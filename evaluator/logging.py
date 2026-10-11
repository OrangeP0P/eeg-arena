import json
import os
from datetime import datetime


def log_metrics(metrics, log_path, extra=None):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    if extra:
        record.update(extra)

    def _default(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_default) + "\n")

    return record
