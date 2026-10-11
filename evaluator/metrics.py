import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    r2_score,
    mean_squared_error,
)


def compute_metrics(y_true, y_pred, task="multiclass"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if task in {"multiclass", "binary"}:
        acc = balanced_accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        f1_macro = f1_score(y_true, y_pred, average="macro")
        kappa = cohen_kappa_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        return {
            "acc": float(acc),
            "kappa": float(kappa),
            "f1_weighted": float(f1_weighted),
            "f1_macro": float(f1_macro),
            "confusion_matrix": cm,
        }

    if task == "regression":
        corrcoef = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size else 0.0
        r2 = float(r2_score(y_true, y_pred)) if y_true.size else 0.0
        rmse = float(mean_squared_error(y_true, y_pred) ** 0.5) if y_true.size else 0.0
        return {"corrcoef": corrcoef, "r2": r2, "rmse": rmse}

    raise ValueError(f"Unknown task: {task}")
