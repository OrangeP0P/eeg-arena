# SKILL: .mat Sidecar Marker Kernel

> **适用模式**: 主数据文件为 `.mat` 格式（或存在同名 `.mat` sidecar），  
> 事件标签存储于 `.mat` 文件的某个字段（如 `marker`、`classlabel`、`event`）中，  
> 而非 MNE Annotations 中。需从 `.mat` 字段提取事件并写回 MNE Annotations。  
> **参考示例**: `kernels/figshare_largemi.py`  
> **规范文档**: `kernels/KERNEL_RULEBOOK.md`

---

## 你的角色

你是一名 EEGUnity kernel 工程师。目标：为给定数据集生成符合 `KERNEL_RULEBOOK.md` 的 Stage1 kernel。  
**在写任何代码前，先完成 Phase 1 全部探索步骤。**

---

## Phase 1: 数据集探索（自动提取）

### 1.1 列出 .mat 文件结构

```bash
find <dataset_path> -name "*.mat" | head -20
```

记录：文件命名规律（如 `CLA-SubjectA-160405-3St-LRHand.mat`、`A01E.mat`）。

### 1.2 读取一个 .mat 文件，检查顶层结构

```python
import scipy.io as sio
mat = sio.loadmat("<sample_mat_file>", simplify_cells=True)
print("顶层 keys:", list(mat.keys()))
# 过滤 '__header__', '__version__', '__globals__'
data_keys = [k for k in mat.keys() if not k.startswith('__')]
print("数据 keys:", data_keys)
```

若顶层有嵌套结构（如 `mat['o']`），继续展开：

```python
inner = mat["o"]
print("内层 keys:", list(inner.keys()) if hasattr(inner, 'keys') else type(inner))
```

### 1.3 定位 marker / label 字段

```python
# 常见字段名：marker, classlabel, event, labels, y
marker = mat["o"]["marker"]   # 根据 1.2 结果替换路径
print("marker shape:", marker.shape)
print("marker dtype:", marker.dtype)
print("unique values:", set(marker.flatten().tolist()[:1000]))
```

记录：
- marker 的 shape（通常为 `(n_samples,)` 或 `(1, n_samples)`）
- 所有唯一值及其含义（查阅数据集文档）
- marker 是"连续标记"（每个采样点一个值）还是"事件索引"（稀疏的时间戳+值对）

**连续标记**：值在整个录制期间逐采样变化，0 表示无事件，非零值表示事件类型。  
**事件索引**：仅列出事件发生时刻的 `[onset_sample, event_code]`，其余时刻没有条目。

### 1.4 读取 EEG 数据结构

```python
# 常见字段：data, eeg, X, signal
eeg = mat["o"]["data"]   # 根据 1.2 结果替换
print("EEG shape:", eeg.shape)  # 期望: (n_channels, n_samples) 或 (n_samples, n_channels)
sfreq = mat["o"]["sampFreq"]    # 或 sFreq, Fs, fs, sample_rate
print("采样率:", sfreq)
ch_names_raw = mat["o"]["chnames"]   # 或 channels, channames
print("通道名:", ch_names_raw)
```

记录：EEG shape、采样率、通道名列表。

### 1.5 提取受试者信息

按优先级尝试：

1. **参与者列表文件**：
   ```bash
   find <dataset_path> -name "participants.tsv" | head -3
   ```
2. **README / 文档**：
   ```bash
   find <dataset_path> -name "README*" -o -name "*.txt" | head -5
   cat <readme_path>
   ```
3. **数据集官方论文**（DOI / URL，若用户提供）。
4. 均无法获取时，所有字段写 `"unknown"`。

记录：每个受试者 ID → age（str）、sex、amplifier、cap、handedness。

### 1.6 确认 marker 类型（连续 vs 事件索引）

- 若 `marker.shape == (n_samples,)` 且大多数值为 0 → **连续标记**（参见模板 Option A）
- 若 marker 是稀疏的事件列表（每行 `[onset, code]`）→ **事件索引**（参见模板 Option B）

---

## Phase 2: Kernel 生成

根据 Phase 1 的提取结果，选择 Option A 或 Option B。

```python
"""<DatasetName> stage1 kernel for EEGUnity.

<简短描述：数据集名称、范式、通道数、事件结构>

Source: <论文 DOI 或数据集 URL>
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import scipy.io as sio


# Source: <论文引用 / 数据集文档 URL>
# Annotator: <claude-model-id>, <YYYY-MM-DD> — LLM 推断，需人类复核
_SUBJECT_DICT: Dict[str, Dict[str, Any]] = {
    # "<sub_key>": {"age": "<age_str>", "sex": "<sex_str>"},
}

# Source: <数据集文档，说明各 marker 值含义>
# Annotator: <claude-model-id>, <YYYY-MM-DD> — LLM 推断，需人类复核
# 格式：{marker_int_value: "annotation_description"}
_EVENT_MAP: Dict[int, str] = {
    # 1: "left hand MI",
    # 2: "right hand MI",
    # ...
}


# ── Option A: 连续标记（每采样点一个 marker 值） ───────────────────────────────

def _build_annotations_from_continuous_marker(
    marker: np.ndarray,
    sfreq: float,
    event_map: Dict[int, str],
) -> mne.Annotations:
    """Convert a per-sample marker array to MNE Annotations.

    Consecutive samples with the same non-zero code are merged into
    a single annotation with the corresponding duration.
    """
    marker = marker.flatten()
    annotations: List[Tuple[float, float, str]] = []
    current_code: Optional[int] = None
    current_start: int = 0

    for i, raw_code in enumerate(marker):
        code = int(raw_code)
        label = event_map.get(code)

        if label is not None:
            if current_code is None:
                current_code = code
                current_start = i
            elif code != current_code:
                annotations.append((
                    current_start / sfreq,
                    (i - current_start) / sfreq,
                    event_map[current_code],
                ))
                current_code = code
                current_start = i
        else:
            if current_code is not None:
                annotations.append((
                    current_start / sfreq,
                    (i - current_start) / sfreq,
                    event_map[current_code],
                ))
                current_code = None

    if current_code is not None:
        annotations.append((
            current_start / sfreq,
            (len(marker) - current_start) / sfreq,
            event_map[current_code],
        ))

    if annotations:
        onset, duration, desc = zip(*annotations)
        return mne.Annotations(list(onset), list(duration), list(desc))
    return mne.Annotations([], [], [])


# ── Option B: 事件索引（稀疏的 [onset_sample, code] 列表） ────────────────────

def _build_annotations_from_event_index(
    events_array: np.ndarray,
    sfreq: float,
    event_map: Dict[int, str],
    default_duration: float = 0.0,
) -> mne.Annotations:
    """Convert a sparse event index array to MNE Annotations.

    events_array: shape (n_events, 2) → columns [onset_sample, event_code]
    default_duration: annotation duration in seconds (0 = instantaneous)
    """
    onsets, durations, descs = [], [], []
    for row in events_array:
        onset_sample, code = int(row[0]), int(row[1])
        label = event_map.get(code)
        if label is not None:
            onsets.append(onset_sample / sfreq)
            durations.append(default_duration)
            descs.append(label)
    return mne.Annotations(onsets, durations, descs)


@dataclass
class <DatasetName>Kernel:
    """Stage1 kernel for the <dataset_name> dataset."""

    KERNEL_ID: str = "<dataset-id>-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]

        # ── Subject ID extraction ────────────────────────────────────────────
        # 根据 Phase 1.1 的文件命名规律调整
        subject_match = re.search(r"<pattern>", os.path.basename(file_path), re.IGNORECASE)
        if subject_match is None:
            raise ValueError(f"Cannot infer subject id from: {file_path!r}")
        subject_key = subject_match.group(1).upper()
        if subject_key not in _SUBJECT_DICT:
            raise ValueError(f"Unknown subject id: {subject_key!r}")

        # ── Step 1: Load .mat file ───────────────────────────────────────────
        mat = sio.loadmat(file_path, simplify_cells=True)
        # 根据 Phase 1.2–1.4 的结果调整路径
        data_struct = mat["<top_key>"]
        marker = data_struct["<marker_key>"]
        sfreq = float(raw.info["sfreq"])

        # ── Step 2: Build annotations ────────────────────────────────────────
        # 选择 Option A 或 Option B，删除另一个
        # Option A（连续标记）:
        annotations = _build_annotations_from_continuous_marker(marker, sfreq, _EVENT_MAP)
        # Option B（事件索引）:
        # annotations = _build_annotations_from_event_index(marker, sfreq, _EVENT_MAP)

        raw.set_annotations(annotations)

        # ── Step 3: Write eegunity_description ──────────────────────────────
        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "age":        _SUBJECT_DICT[subject_key]["age"],
                "sex":        _SUBJECT_DICT[subject_key]["sex"],
                "amplifier":  "<amplifier_str>",
                "cap":        "<cap_str>",
                "handedness": "<handedness_str>",
            },
        }
        raw.info["description"] = json.dumps(description_dict)
        return raw


KERNEL = <DatasetName>Kernel()
```

---

## Phase 3: 验证清单（对照 KERNEL_RULEBOOK.md）

| 编号 | 规则 | 检查项 |
|------|------|--------|
| 第5条 | 接口 | 模块末尾有 `KERNEL = <ClassName>()` |
| 第5条 | 接口 | `apply(self, udataset, raw, row) -> mne.io.BaseRaw` 签名正确 |
| 第8条 | 描述 | `raw.info["description"]` 写入合法 JSON，包含 `original_description` + `eegunity_description` |
| 第9条 | 描述 | `eegunity_description` 包含全部 5 个字段：`age`、`sex`、`amplifier`、`cap`、`handedness` |
| 第10条 | age | `age` 为字符串类型（`"22"` 而非整数 `22`） |
| 第11条 | sex | `sex` 为小写 `"male"`/`"female"`/`"unknown"`（不得有大写首字母） |
| 第11条 | 来源 | `_SUBJECT_DICT` 有来源注释；LLM 填写须标注"需人类复核" |
| 第12条 | 标签 | `_EVENT_MAP` 中的描述字符串为人类可读，而非裸数字 |
| 第13条 | raw meta | 所有 annotation 均对应原始 marker 值，未引入语义派生 |
| 第16条 | 外部文件 | `.mat` 读取逻辑有合理的错误提示（找不到字段时 raise 明确信息） |
| 第17条 | 注释 | `_EVENT_MAP` 有来源注释（论文/文档 + 标注者 + 日期） |
| 第3条 | LLM诚信 | LLM 推断的信息全部标注"需人类复核" |
