"""
Unified Trainer - Compatible with any EEG model
"""
import copy
import os
import statistics
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from tqdm import tqdm

from evaluator import collect_outputs, evaluate_embeddings
from experiments.common.tracking import now_iso, scalarize_metrics


class Trainer:
    """
    Unified Trainer class

    Args:
        model: PyTorch model
        data_loader: dict containing train/val/test
        params: training parameters (epochs, lr, batch_size, etc.)
    """

    # Task type to loss function mapping
    TASK_CRITERIONS = {
        'multiclass': CrossEntropyLoss,
        'binary': BCEWithLogitsLoss,
        'regression': MSELoss,
    }

    # Dataset to task type mapping (can be extended)
    DATASET_TASK_MAP = {
        'bcic_iv_1': 'multiclass',
        'figshare_shudb': 'multiclass',
        'physionet_eegmmidb': 'multiclass',
        'bcic_iv_2b': 'multiclass',
        'figshare_largemi': 'multiclass',
        'seed_100': 'multiclass',
        'other_highgammadataset': 'multiclass',
        'bcic_iv_2a': 'multiclass',
        'eegmat': 'multiclass',
        'MentalArithmetic': 'binary',
        'TUAB': 'binary',
        'SEED-VIG': 'regression',
    }

    def _detect_task_type(self, params, data_loader):
        """Automatically detect task type based on num_classes and label type"""
        num_classes = getattr(params, 'num_of_classes', None)

        # If num_classes is provided, use it
        if num_classes is not None:
            if num_classes == 2:
                return 'binary'
            elif num_classes > 2:
                return 'multiclass'

        # Fallback: check label type from first batch
        try:
            _, labels = next(iter(data_loader['train']))
            if labels.dtype in [torch.float32, torch.float64]:
                return 'regression'
        except Exception:
            pass

        # Default to multiclass
        return 'multiclass'

    def __init__(self, model, data_loader, params, tracker=None):
        self.device = getattr(params, "device_obj", None)
        if self.device is None:
            default_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(getattr(params, "device", default_device))

        self.params = params
        self.tracker = tracker
        self.use_dataparallel = getattr(params, "use_dataparallel", False)
        self.gpu_ids = None

        # Wrap model with DataParallel if requested
        if self.use_dataparallel and self.device.type == "cuda":
            gpu_ids_str = getattr(params, "gpu_ids", "0")
            self.gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(',')]
            if len(self.gpu_ids) > 1:
                print(f"Using DataParallel on GPUs: {self.gpu_ids}")
                # First move model to first GPU, then wrap with DataParallel
                first_gpu = self.gpu_ids[0]
                model = model.cuda(first_gpu)
                self.model = torch.nn.DataParallel(model, device_ids=self.gpu_ids)
                self.device = torch.device(f"cuda:{first_gpu}")
            else:
                self.model = model.to(self.device)
        else:
            self.model = model.to(self.device)

        self.data_loader = data_loader

        # Auto-detect task type
        task_type = self._detect_task_type(params, data_loader)
        self.task_type = task_type

        # Setup loss function
        # Both binary and multiclass use CrossEntropyLoss (model outputs [batch, num_classes])
        if task_type in ['binary', 'multiclass']:
            self.criterion = CrossEntropyLoss(
                label_smoothing=getattr(params, 'label_smoothing', 0.0)
            ).to(self.device)
        elif task_type == 'regression':
            self.criterion = MSELoss().to(self.device)

        self.best_model_states = None
        self.best_epoch = 0
        self.best_metric_value = -float("inf")
        self.best_val_metrics = None
        self.selection_metric_name = self._selection_metric_name()
        self.early_stopping_patience = int(getattr(params, "early_stopping_patience", 10))
        self.epochs_without_improvement = 0

        # Setup optimizer
        self._setup_optimizer()

        # Learning rate scheduler
        self.data_length = len(data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.params.epochs * self.data_length,
            eta_min=1e-6
        )

    def _model_for_custom_interfaces(self):
        """
        Return the underlying model object that owns optional custom APIs.

        DataParallel wraps the module and does not reliably expose project-
        specific helpers like freeze_backbone() or get_param_groups().
        """
        if self.use_dataparallel and hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def _setup_optimizer(self):
        """Setup optimizer"""
        lr = getattr(self.params, 'lr', 1e-4)
        wd = getattr(self.params, 'weight_decay', 5e-2)
        optimizer_name = getattr(self.params, 'optimizer', 'AdamW')
        interface_model = self._model_for_custom_interfaces()

        if getattr(self.params, 'frozen', False):
            if hasattr(interface_model, "freeze_backbone") and callable(interface_model.freeze_backbone):
                interface_model.freeze_backbone()
            else:
                print("Warning: --frozen is set, but model has no freeze_backbone(). Skipping freeze.")

        if getattr(self.params, 'multi_lr', True):
            param_groups = self._resolve_param_groups(lr=lr, wd=wd)
            if optimizer_name == 'AdamW':
                self.optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
            else:
                self.optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=wd)
        else:
            # Use unified learning rate on all trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable_params:
                raise ValueError("No trainable parameters found after freeze/filtering.")

            if optimizer_name == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    trainable_params, lr=lr, weight_decay=wd
                )
            else:
                self.optimizer = torch.optim.SGD(
                    trainable_params, lr=lr, momentum=0.9, weight_decay=wd
                )

    def _resolve_param_groups(self, lr, wd):
        """
        Resolve optimizer param groups.
        Priority:
        1) model.get_param_groups() if provided by model
        2) fallback to a single trainable-param group
        """
        interface_model = self._model_for_custom_interfaces()

        if hasattr(interface_model, "get_param_groups") and callable(interface_model.get_param_groups):
            raw_groups = interface_model.get_param_groups()
            if not isinstance(raw_groups, (list, tuple)):
                raise TypeError("model.get_param_groups() must return a list/tuple of dicts.")

            resolved_groups = []
            for group in raw_groups:
                if not isinstance(group, dict) or "params" not in group:
                    raise TypeError("Each param group must be a dict containing 'params'.")

                params = [p for p in group["params"] if p.requires_grad]
                if not params:
                    continue

                resolved_group = {"params": params}
                lr_scale = group.get("lr_scale", 1.0)
                for key, value in group.items():
                    if key in {"params", "lr_scale"}:
                        continue
                    resolved_group[key] = value

                if "lr" not in resolved_group:
                    resolved_group["lr"] = lr * lr_scale
                resolved_group.setdefault("weight_decay", wd)
                resolved_groups.append(resolved_group)

            if not resolved_groups:
                raise ValueError("model.get_param_groups() returned no trainable parameters.")
            return resolved_groups

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found for optimizer.")
        return [{"params": trainable_params, "lr": lr, "weight_decay": wd}]

    def _compute_loss(self, pred, target):
        """Allow models to optionally augment the default task loss."""
        interface_model = self._model_for_custom_interfaces()
        if hasattr(interface_model, "compute_loss") and callable(interface_model.compute_loss):
            return interface_model.compute_loss(pred, target, self.criterion)

        if getattr(self.params, 'downstream_dataset', None) == 'ISRUC':
            return self.criterion(pred.transpose(1, 2), target)
        return self.criterion(pred, target)

    def _train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        start_time = timer()
        losses = []

        profile_steps_value = os.environ.get(
            "EEG_ARENA_PROFILE_TRAIN_STEPS",
            os.environ.get("EEG2_PROFILE_TRAIN_STEPS", ""),
        )
        profile_steps = profile_steps_value.strip().lower() in {
            "1", "true", "yes", "on"
        }
        step_times = []
        if profile_steps and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

        for step_index, (x, y) in enumerate(tqdm(self.data_loader['train'], mininterval=10), start=1):
            if profile_steps and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            step_started = timer()
            self.optimizer.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)
            loss = self._compute_loss(pred, y)

            loss.backward()
            losses.append(loss.data.cpu().numpy())

            # Gradient clipping
            if getattr(self.params, 'clip_value', 1) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.params.clip_value
                )

            self.optimizer.step()
            self.optimizer_scheduler.step()

            if profile_steps:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                step_sec = timer() - step_started
                step_times.append(step_sec)
                peak_allocated_mib = (
                    torch.cuda.max_memory_allocated(self.device) / 2**20
                    if self.device.type == "cuda"
                    else 0.0
                )
                peak_reserved_mib = (
                    torch.cuda.max_memory_reserved(self.device) / 2**20
                    if self.device.type == "cuda"
                    else 0.0
                )
                print(
                    "[train-step-profile] "
                    f"epoch={epoch + 1} step={step_index} seconds={step_sec:.6f} "
                    f"peak_cuda_allocated_mib={peak_allocated_mib:.2f} "
                    f"peak_cuda_reserved_mib={peak_reserved_mib:.2f}",
                    flush=True,
                )

        if profile_steps and step_times:
            steady_times = step_times[1:] or step_times
            print(
                "[train-step-profile-summary] "
                f"epoch={epoch + 1} steps={len(step_times)} "
                f"first_step_seconds={step_times[0]:.6f} "
                f"steady_median_seconds={statistics.median(steady_times):.6f}",
                flush=True,
            )

        return np.mean(losses), (timer() - start_time) / 60

    def _evaluate(self, split='val'):
        """Evaluate model"""
        emb, logits, labels = collect_outputs(
            self.model, self.data_loader[split]
        )
        metrics = evaluate_embeddings(
            emb, labels, logits=logits, task=self.task_type, log_path=None
        )
        return metrics

    def _selection_metric_name(self):
        if self.task_type == 'multiclass':
            return 'kappa'
        if self.task_type == 'binary':
            return 'acc'
        if self.task_type == 'regression':
            return 'r2'
        raise ValueError(f"Unsupported task type: {self.task_type}")

    def _selection_metric_value(self, metrics):
        return float(metrics[self.selection_metric_name])

    def _current_lr_groups(self):
        return [float(group.get("lr", 0.0)) for group in self.optimizer.param_groups]

    def _should_save_checkpoints(self):
        return bool(getattr(self.params, "save_checkpoints", True))

    def _save_checkpoint(self, path):
        if self.use_dataparallel and hasattr(self.model, 'module'):
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def _snapshot_model_state(self):
        """Keep the best checkpoint off-device to avoid duplicating GPU memory."""
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }

    def _update_best_state(self, val_metrics, epoch, selection_metric_value):
        self.best_model_states = self._snapshot_model_state()
        self.best_epoch = epoch
        self.best_metric_value = selection_metric_value
        self.best_val_metrics = copy.deepcopy(val_metrics)

        if self.tracker is not None:
            self.tracker.write_best_val_metrics(
                {
                    "timestamp": now_iso(),
                    "epoch": epoch,
                    "selection_metric_name": self.selection_metric_name,
                    "selection_metric_value": selection_metric_value,
                    "metrics": val_metrics,
                }
            )
            self.tracker.update_status(
                best_epoch=epoch,
                selection_metric_name=self.selection_metric_name,
                selection_metric_value=selection_metric_value,
                best_checkpoint_path=self.tracker.best_checkpoint_path,
            )

    def _record_epoch_metrics(self, epoch, loss, time_elapsed, val_metrics, test_metrics, is_best):
        if self.tracker is None:
            return

        selection_metric_value = self._selection_metric_value(val_metrics)
        self.tracker.append_epoch_metrics(
            {
                "timestamp": now_iso(),
                "epoch": epoch,
                "train_loss": float(loss),
                "train_minutes": float(time_elapsed),
                "lr_groups": self._current_lr_groups(),
                "val_metrics": scalarize_metrics(val_metrics),
                "test_metrics": scalarize_metrics(test_metrics),
                "selection_metric_name": self.selection_metric_name,
                "selection_metric_value": selection_metric_value,
                "is_best": bool(is_best),
            }
        )

    def _print_epoch_summary(self, epoch, loss, time_elapsed, val_metrics, test_metrics):
        if self.task_type == 'regression':
            print(
                f"Epoch {epoch}: Loss={loss:.5f}, "
                f"val_corrcoef={val_metrics['corrcoef']:.5f}, val_r2={val_metrics['r2']:.5f}, val_rmse={val_metrics['rmse']:.5f}, "
                f"test_r2={test_metrics['r2']:.5f}, Time={time_elapsed:.2f}min"
            )
            return

        print(
            f"Epoch {epoch}: Loss={loss:.5f}, "
            f"val_acc={val_metrics['acc']:.5f}, val_kappa={val_metrics['kappa']:.5f}, "
            f"val_f1_w={val_metrics['f1_weighted']:.5f}, val_f1_m={val_metrics['f1_macro']:.5f}, "
            f"test_acc={test_metrics['acc']:.5f}, test_kappa={test_metrics['kappa']:.5f}, "
            f"Time={time_elapsed:.2f}min"
        )

    def _restore_best_state(self):
        """Restore best model if available; otherwise keep current parameters."""
        if self.best_model_states is not None:
            self.model.load_state_dict(self.best_model_states)

    def _should_early_stop(self):
        return (
            self.early_stopping_patience > 0
            and self.epochs_without_improvement >= self.early_stopping_patience
        )

    def train(self):
        """
        Unified training entry - automatically selects training method based on task type
        """
        for epoch in range(1, self.params.epochs + 1):
            loss, time_elapsed = self._train_epoch(epoch - 1)
            val_metrics = self._evaluate('val')
            test_metrics = self._evaluate('test')

            selection_metric_value = self._selection_metric_value(val_metrics)
            is_best = selection_metric_value > self.best_metric_value

            self._print_epoch_summary(epoch, loss, time_elapsed, val_metrics, test_metrics)
            self._record_epoch_metrics(epoch, loss, time_elapsed, val_metrics, test_metrics, is_best)

            if is_best:
                print(f"  -> {self.selection_metric_name} improved, tracking best checkpoint state.")
                self._update_best_state(val_metrics, epoch, selection_metric_value)
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self._should_early_stop():
                print(
                    f"  -> Early stopping triggered: no improvement in val_{self.selection_metric_name} "
                    f"for {self.epochs_without_improvement} consecutive epochs."
                )
                if self.tracker is not None:
                    self.tracker.update_status(
                        stopped_early=True,
                        stopped_epoch=epoch,
                        early_stopping_patience=self.early_stopping_patience,
                    )
                break

        return self._finalize_training()

    def _finalize_training(self):
        if self.best_model_states is None:
            self.best_model_states = self._snapshot_model_state()

        if self.tracker is not None and self._should_save_checkpoints():
            self._save_checkpoint(self.tracker.last_checkpoint_path)
            self.tracker.update_status(last_checkpoint_path=self.tracker.last_checkpoint_path)
        elif self.tracker is not None:
            print("Checkpoint saving disabled; skipping last/best checkpoint files.")
            self.tracker.update_status(last_checkpoint_path=None, best_checkpoint_path=None)

        self._restore_best_state()

        if self.tracker is not None and self._should_save_checkpoints():
            self._save_checkpoint(self.tracker.best_checkpoint_path)
            self.tracker.update_status(best_checkpoint_path=self.tracker.best_checkpoint_path)

        print("=" * 50)
        print("Testing best checkpoint...")
        final_test_metrics = self._evaluate('test')

        if self.tracker is not None:
            self.tracker.write_final_test_metrics(
                {
                    "timestamp": now_iso(),
                    "epoch": self.best_epoch,
                    "selection_metric_name": self.selection_metric_name,
                    "selection_metric_value": self.best_metric_value,
                    "metrics": final_test_metrics,
                }
            )
            self.tracker.update_status(
                best_epoch=self.best_epoch,
                selection_metric_name=self.selection_metric_name,
                selection_metric_value=self.best_metric_value,
                stopped_early=self._should_early_stop(),
                early_stopping_patience=(
                    self.early_stopping_patience if self.early_stopping_patience > 0 else None
                ),
                final_test_metrics_path=self.tracker.final_test_metrics_path,
            )

        if self.task_type == 'regression':
            print(
                f"Final test: corrcoef={final_test_metrics.get('corrcoef', 0):.5f}, "
                f"r2={final_test_metrics.get('r2', 0):.5f}, rmse={final_test_metrics.get('rmse', 0):.5f}"
            )
        else:
            print(
                f"Final test: acc={final_test_metrics.get('acc', 0):.5f}, "
                f"kappa={final_test_metrics.get('kappa', 0):.5f}, "
                f"f1_w={final_test_metrics.get('f1_weighted', 0):.5f}, "
                f"f1_m={final_test_metrics.get('f1_macro', 0):.5f}"
            )

        return final_test_metrics
