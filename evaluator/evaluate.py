import numpy as np

from .metrics import compute_metrics
from .logging import log_metrics


def _pred_from_logits(logits):
    logits = np.asarray(logits)
    if logits.ndim == 1:
        return (logits > 0).astype(int)
    if logits.ndim == 2:
        return np.argmax(logits, axis=1)
    return logits.squeeze()


def evaluate_embeddings(embeddings, labels, logits=None, task="multiclass", log_path=None, extra=None):
    y_true = np.asarray(labels)
    if logits is not None:
        y_pred = _pred_from_logits(logits)
    else:
        y_pred = np.asarray(embeddings).squeeze()

    metrics_result = compute_metrics(y_true, y_pred, task=task)

    if log_path:
        log_metrics(metrics_result, log_path, extra=extra)

    return metrics_result
