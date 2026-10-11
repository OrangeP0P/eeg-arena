# EEG Model Integration Guide (Unified Interfaces)

This guide describes how to add a new EEG model for **fine-tuning** in the current framework.

## 1. Unified Contracts (Must Follow)

All new models, adapters, and configs should follow these contracts:

| Component | Contract |
|---|---|
| ModelAdapter | `Adapter(args)` + `get_data_loader()` |
| Model | `forward(x)` **must** return logits; optional `extract_features(x)` |
| Config | `add_args(parser, mode)`, `get_model_keys()`, `get_input_constraints()` |
| Optimizer Groups | Prefer `model.get_param_groups()`; optional `model.freeze_backbone()` |

Why this matters:

- New models can be added without changing `run_single_experiment.py`.
- Evaluator and Trainer no longer depend on model-internal naming conventions.
- Weight resolution, input constraints, and registry lookups all happen through shared infrastructure.

## 2. Fine-tuning Flow In The Current Repo

```text
run_single_experiment.py
  -> parse common args
  -> parse model-specific args via config.add_args(parser, mode='finetune')
  -> apply input constraints from config.get_input_constraints()
  -> handle --list_variants / resolve --variant -> --foundation_dir
  -> models.registry.get_finetune_adapter(model)
  -> adapter = FinetuneDataLoader(args)
  -> data_loader = adapter.get_data_loader()
  -> model = Model(args)
  -> Trainer(model, data_loader, args)
```

Important runtime behaviors:

- `run_single_experiment.py` injects model-specific CLI args dynamically after it sees `--model`.
- `experiments/single/input_constraints.py` enforces `required_input_format` and `required_patch_size`.
- `experiments/single/weight_resolution.py` resolves `--variant` into a real `foundation_dir` before model construction.
- `FinetuneDataLoader` first respects the user-selected `--channel_system`; if the model declares a dedicated channel space in `data/channels.py`, the data path applies an additional channel-space alignment step.
- `channel_system=custom` is reserved for explicit channel-sparsity experiments. In this mode, `--channel_variant` selects a named channel list from a JSON config before any model-specific channel-space intersection is applied.
- If a user explicitly passes a conflicting input constraint, the run fails fast.

## 3. Recommended Project Structure

```text
models/NEW_MODEL/
├── __init__.py              # Register model to framework
├── config.py                # Configuration class
└── models/
    ├── __init__.py          # Export components (optional)
    ├── <your_module_name>.py # Original/model-specific implementation
    ├── finetune_model.py    # Wrapper with unified interface
    └── ...                  # Other model components

models/Weights/
└── NEW_MODEL/
    └── model.pth            # Pretrained weights (optional)
```

If your model uses a single file rather than a directory, it is also fine to use a direct path such as:

```text
models/Weights/NEW_MODEL/model.pth
```

For scratch baselines such as `EEGNet` or `DeepConvNet`, the `models/Weights/NEW_MODEL/` directory can be omitted entirely.

The source module name inside `models/NEW_MODEL/models/` is not a framework contract.
You can keep the upstream/original filename or choose a descriptive local name such as
`backbone.py`, `encoder.py`, or `upstream.py`, as long as `finetune_model.py` imports
from the real module path.

## 4. Wrapper Template (`models/NEW_MODEL/models/finetune_model.py`)

This wrapper adapts your original model to the unified framework interface.

```python
import os
import torch
import torch.nn as nn

from .<your_module_name> import YourOriginalModel


class Model(nn.Module):
    """
    Unified model contract:
    - forward(x) returns logits [B, num_classes]
    - extract_features(x) is optional but recommended
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

        # Dataset-provided metadata (auto-filled by FinetuneDataLoader)
        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        patch_num = getattr(params, "patch_num", 30)
        ch_positions = getattr(params, "ch_positions", None)
        ch_names = getattr(params, "ch_names", None)
        model_channel_indices = getattr(params, "model_channel_indices", None)

        dropout = getattr(params, "dropout", 0.1)

        self.backbone = YourOriginalModel(
            in_channels=num_channels,
            dropout=dropout,
        )

        if getattr(params, "use_pretrained_weights", False):
            foundation_dir = getattr(params, "foundation_dir", "./models/Weights/NEW_MODEL")

            # Support both direct file paths and directory paths.
            if os.path.isfile(foundation_dir):
                pretrained_file = foundation_dir
            else:
                pretrained_file = os.path.join(foundation_dir, "model.pth")

            if os.path.exists(pretrained_file):
                checkpoint = torch.load(pretrained_file, map_location="cpu", weights_only=False)

                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

                msg = self.backbone.load_state_dict(state_dict, strict=False)
                print(f"[NEW_MODEL] Loaded pretrained weights from {pretrained_file}")
                if msg.missing_keys:
                    print(f"[NEW_MODEL] Missing keys: {msg.missing_keys[:5]}...")
                if msg.unexpected_keys:
                    print(f"[NEW_MODEL] Unexpected keys: {msg.unexpected_keys[:5]}...")
            else:
                print(f"[NEW_MODEL] Warning: Pretrained weights not found at {pretrained_file}")

        if getattr(params, "frozen", False):
            self.freeze_backbone()

        embed_dim = 200  # Adjust to match your model
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def extract_features(self, x):
        """Optional API used by the evaluator for embeddings."""
        return self.backbone(x)

    def forward(self, x):
        """Required: returns logits [B, num_classes]."""
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits

    def freeze_backbone(self):
        """Optional API used when --frozen is enabled."""
        for name, param in self.backbone.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        print("[NEW_MODEL] Backbone frozen (except classification head)")

    def get_param_groups(self):
        """
        Optional API for optimizer parameter grouping.
        Trainer prioritizes this over fallback grouping.
        """
        backbone_params = []
        head_params = []

        for name, param in self.backbone.named_parameters():
            if "head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr_scale": 1.0},
            {"params": head_params, "lr_scale": 10.0},
        ]
```

Important notes:

- `forward()` must return logits of shape `(B, num_classes)`.
- If `extract_features()` is missing, the evaluator falls back to logits.
- Use `getattr(params, ..., default)` for optional parameters to avoid `AttributeError`.
- `<your_module_name>.py` is only an example placeholder, not a required filename.
- Your wrapper should not implement variant resolution itself; by the time `Model(params)` runs, `foundation_dir` has already been resolved by the entrypoint when variants are used.

## 5. Config Template (`models/NEW_MODEL/config.py`)

```python
import argparse
from dataclasses import dataclass


@dataclass
class NewModelFinetuneDefaults:
    """Defaults for IDE auto-completion only."""
    dropout: float = 0.1
    frozen: bool = False
    use_pretrained_weights: bool = True
    foundation_dir: str = "./models/Weights/NEW_MODEL"


class NewModelConfig:
    """NewModel configuration class."""

    MODEL_KEYS = ["dropout"]

    # Optional: only define these if your model supports multiple official checkpoints.
    WEIGHT_VARIANTS = {
        "base": {
            "file": "model-base.pth",
            "description": "Base pretrained checkpoint",
        },
        "large": {
            "file": "model-large.pth",
            "description": "Large pretrained checkpoint",
        },
    }
    DEFAULT_VARIANT = "base"
    WEIGHTS_DIR = "./models/Weights/NEW_MODEL"

    @staticmethod
    def get_model_keys():
        return NewModelConfig.MODEL_KEYS.copy()

    @staticmethod
    def get_input_constraints():
        return {
            "required_input_format": "patched",  # or "raw"
            "required_patch_size": 200,           # omit if not fixed
            "supports_pretrained": True,
        }

    @staticmethod
    def add_args(parser, mode="all"):
        if mode in ["all", "finetune"]:
            NewModelConfig._add_finetune_args(parser)

    @staticmethod
    def _add_finetune_args(parser):
        group = parser.add_argument_group("NewModel Fine-tune options")

        group.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Dropout rate (default: 0.1)",
        )
        group.add_argument(
            "--foundation_dir",
            type=str,
            default="./models/Weights/NEW_MODEL",
            help="Path to pretrained weights file or directory",
        )
        group.add_argument(
            "--variant",
            type=str,
            default=None,
            choices=list(NewModelConfig.WEIGHT_VARIANTS.keys()),
            help="Optional pretrained-weight variant",
        )
```

Important:

- Do not use `type=bool` for CLI booleans.
- Use `argparse.BooleanOptionalAction`.
- `--frozen` and `--use_pretrained_weights` are already provided by `run_single_experiment.py`; wrappers can consume them with `getattr(params, ...)`, but model configs should not re-declare them.
- `get_input_constraints()` controls the automatic runtime enforcement of input-format constraints.
- Only define `WEIGHT_VARIANTS`, `DEFAULT_VARIANT`, and `WEIGHTS_DIR` if your model truly supports multiple named checkpoints.

## 6. Do Not Re-Declare Common CLI Args

`run_single_experiment.py` already owns the common runtime CLI surface. In model configs, do **not** re-declare these shared flags:

- `--model`
- `--use_pretrained_weights`
- `--frozen`
- `--seed`
- `--device`
- `--cuda`
- `--gpu_ids`
- `--use_dataparallel`
- `--epochs`
- `--batch_size`
- `--lr`
- `--weight_decay`
- `--optimizer`
- `--clip_value`
- `--num_workers`
- `--dataset`
- `--task`
- `--cache_dir`
- `--split_mode`
- `--fold`
- `--held_out_subject`
- `--input_format`
- `--patch_size`
- `--channel_system`
- `--output_dir`
- `--save_checkpoints`
- `--label_smoothing`
- `--multi_lr`
- `--encoder_lr_scale`
- `--classifier_lr_scale`
- `--early_stopping_patience`
- `--list_variants`
- `--pipeline`
- `--pipeline_param`
- `--epoch_param`
- `--rebuild_hdf5`

In practice, model configs should only define model-specific knobs plus optional pretrained-weight flags.

## 7. Scratch Baseline Variant

For models that are trained from scratch and do not support checkpoint loading, keep the same registration structure but omit pretrained-related CLI flags entirely.

Typical differences from the pretrained template:

- `supports_pretrained` should be `False`
- do not re-declare `--frozen` or `--use_pretrained_weights`
- do not add `--foundation_dir`
- do not add `--variant`
- use `required_input_format="raw"` when the model consumes `(B, C, T)` EEG directly

This is exactly how the current `EEGNet`, `DeepConvNet`, and `EEGConformer` wrappers are organized.

## 8. Model Registration (`models/NEW_MODEL/__init__.py`)

```python
"""NewModel package."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.registry import register_model
from .config import NewModelConfig
from .models.finetune_model import Model as NewModelFinetune
from data.finetune import FinetuneDataLoader

register_model(
    name="newmodel",
    model_class=NewModelFinetune,
    config_class=NewModelConfig,
    dataset_adapter_class=FinetuneDataLoader,
)

__all__ = ["NewModelFinetune", "NewModelConfig"]
```

Notes:

- The `name=` value is the exact string users pass to `--model`.
- For standard downstream fine-tuning, reuse `data.finetune.FinetuneDataLoader` as the dataset adapter.

## 9. Running Fine-Tuning

Basic run:

```bash
python run_single_experiment.py \
  --model newmodel \
  --dataset bcic_iv_2b \
  --task motor_imagery \
  --cache_dir /path/to/cache \
  --pipeline datasets/pipelines/classical_pipeline_with_z_score.py \
  --pipeline_param resample=200 lower_bound=0.1 upper_bound=75 notch=50 \
  --split_mode loso \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-4
```

Query variants:

```bash
python run_single_experiment.py \
  --model newmodel \
  --dataset bcic_iv_2b \
  --task motor_imagery \
  --cache_dir /path/to/cache \
  --list_variants
```

Run with a named variant:

```bash
python run_single_experiment.py \
  --model newmodel \
  --dataset bcic_iv_2b \
  --task motor_imagery \
  --cache_dir /path/to/cache \
  --variant base
```

Input format reminders:

- `patched`: `(B, C, patch_num, patch_size)`
- `raw`: `(B, C, T)`

## 10. FinetuneDataLoader: Data And Metadata

The framework provides a unified `FinetuneDataLoader` that handles data loading, transformation, and batching.

### 10.1 Return Value of `get_data_loader()`

```python
from data.finetune import FinetuneDataLoader

data_loader = FinetuneDataLoader(args).get_data_loader()
```

Returns a dictionary with the following structure:

| Key | Type | Description |
|---|---|---|
| `'train'` | `DataLoader` | Training data loader |
| `'val'` | `DataLoader` | Validation data loader |
| `'test'` | `DataLoader` | Test data loader |
| `'num_classes'` | `int` | Number of classification classes |
| `'num_channels'` | `int` | Number of EEG channels |
| `'patch_num'` | `int` | Number of patches |
| `'ch_positions'` | `dict` | Channel positions `{channel_name: [x, y, z]}` |
| `'ch_names'` | `list` | Final aligned channel names after channel-system/model-space filtering |
| `'model_channel_indices'` | `list` or `None` | Indices into the declared model channel space, or `None` if no dedicated model space is declared |

### 10.2 Data Format

The data loader supports two input formats:

| Format | Shape | Description |
|---|---|---|
| `patched` | `(B, C, patch_num, patch_size)` | Pre-patched EEG |
| `raw` | `(B, C, T)` | Raw EEG |

### 10.3 Model-Specific Channel Spaces

The shared channel-space configuration lives in `data/channels.py`:

- `CHANNEL_SYSTEMS`: user-selectable channel systems such as `10-20`, `10-10`, and `10-5`
- `MODEL_CHANNEL_SPACES`: optional model-specific pretrained channel spaces
- `get_model_channel_space(model_name, variant=None)`: lookup helper used by `FinetuneDataLoader`; `variant` is used by models such as BIOT (16/18-channel tables) and Signal-JEPA (`full` fixed vocabulary versus `without_channels` montage-free)

Use this mechanism only when your pretrained weights are tied to a fixed electrode layout, fixed channel ordering, or fixed positional-embedding slots. If your model can consume the user-selected channel system directly, do not register a model-specific channel space.

Alignment rules:

- If a model is not registered in `MODEL_CHANNEL_SPACES`, the data loader uses the user-selected `--channel_system` directly.
- If a model is registered, the data path uses the intersection of the user-selected channel system and the model's pretrained channel space.
- If `--channel_system custom` is used, the user-selected channel system is resolved from `--channel_variant`; by default this reads `experiments/configs/table_v/table_v_channel_config.json`, unless `--channel_sparsity_config` points to another JSON file.
- `channel_count_target` records the user-space variant size before model-space intersection.
- `channel_count_loaded` records the final number of channels after dataset/cache validation and model-space intersection.
- If a model has variant-aware channel spaces, `FinetuneDataLoader` passes `args.variant` to `get_model_channel_space()`. BIOT defaults to the `shhs-prest-18ch` space; Signal-JEPA defaults to `full`, while `without_channels` deliberately returns no fixed model space.
- The final channel order follows the model channel space order, not the dataset's original channel order.
- `ch_names` contains that final aligned channel list.
- `model_channel_indices` contains the index of each retained channel inside the declared model channel space.
- If no model-specific channel space is declared, `model_channel_indices` is `None`.
- This alignment step does not synthesize missing model channels, interpolate channels, derive bipolar signals, or add BENDR's relative-amplitude feature; those remain model/pipeline-specific work.

Minimal registration template:

```python
# data/channels.py

MODEL_CHANNEL_SPACE_NEWMODEL = [
    "EEG:Fp1",
    "EEG:Fz",
    "EEG:Cz",
    "EEG:Pz",
]

MODEL_CHANNEL_SPACES = {
    "newmodel": MODEL_CHANNEL_SPACE_NEWMODEL,  # use the registry name passed to --model
}
```

Typical consumer patterns:

| Model Pattern | What It Uses | Why |
|---|---|---|
| `EEGPT` | filtered `ch_names` | The wrapper expects the final channel list to already match EEGPT's pretrained channel dictionary |
| `LaBraM` | `model_channel_indices` | The wrapper maps retained channels back into pretrained positional-embedding slots |

### 10.4 Metadata Auto-Filled To `args`

After `get_data_loader()`, the framework automatically populates `args` with dataset metadata:

```python
args.num_channels         # Number of EEG channels
args.patch_num            # Number of patches
args.num_of_classes       # Number of classes
args.ch_positions         # {channel_name: [x, y, z]}
args.ch_names             # Final aligned channel names after filtering/alignment
args.model_channel_indices  # Indices into the model channel space, or None
```

### 10.5 Using Metadata In Your Model

Access these values in your model `__init__` with `getattr()`:

```python
class Model(nn.Module):
    def __init__(self, params):
        super().__init__()

        num_classes = getattr(params, "num_of_classes", 2)
        num_channels = getattr(params, "num_channels", 64)
        patch_num = getattr(params, "patch_num", 30)

        ch_positions = getattr(params, "ch_positions", None)
        ch_names = getattr(params, "ch_names", None)
        model_channel_indices = getattr(params, "model_channel_indices", None)
```

### 10.6 Use Cases For Metadata

| Metadata | Use Case |
|---|---|
| `num_channels` | Initialize channel-dependent layers |
| `patch_num` | Calculate feature sizes for classifier heads |
| `ch_positions` | Spatial-aware models such as REVE |
| `ch_names` | Final aligned channel list for models such as CSBrain or EEGPT |
| `model_channel_indices` | Positional-index lookup for models such as LaBraM |
| `num_of_classes` | Initialize classifier output layers |
