import numpy as np
import torch


def collect_outputs(model, data_loader, device=None):
    """
    Collect model outputs: embeddings, logits and labels.

    Args:
        model: PyTorch model
        data_loader: DataLoader instance
        device: Computing device (default uses model's device)

    Returns:
        embeddings: Model features (flattened)
        logits: Model output logits
        labels: Ground truth labels
    """
    model.eval()
    embeddings = []
    logits = []
    labels = []

    if device is None:
        device = next(model.parameters()).device

    # DataParallel wraps the module and may not expose project-specific
    # helper APIs like extract_features() directly.
    feature_model = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            # Unified model contract: forward(x) must return logits.
            out = model(x)
            if not isinstance(out, torch.Tensor):
                raise TypeError(
                    f"Model {type(model)} forward() must return logits as torch.Tensor, got {type(out)}."
                )

            # Optional feature API for evaluation/visualization.
            if hasattr(feature_model, "extract_features") and callable(feature_model.extract_features):
                feats = feature_model.extract_features(x)
                if not isinstance(feats, torch.Tensor):
                    raise TypeError(
                        f"Model {type(feature_model)} extract_features() must return torch.Tensor, got {type(feats)}."
                    )
                feats_flat = feats.reshape(feats.shape[0], -1)
            else:
                feats_flat = out.reshape(out.shape[0], -1)

            embeddings.append(feats_flat.detach().cpu().numpy())
            logits.append(out.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0))
    logits = np.concatenate(logits, axis=0) if logits else np.empty((0, 0))
    labels = np.concatenate(labels, axis=0) if labels else np.empty((0,))

    return embeddings, logits, labels
