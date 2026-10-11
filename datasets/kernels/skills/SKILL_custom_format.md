# SKILL: Custom / Proprietary Format Kernel (HANDLES_UNAVAILABLE)

> **适用模式**: EEGUnity 无法直接解析数据文件（非标准格式、私有二进制、  
> 多文件组合等），需在 kernel 中实现 `HANDLES_UNAVAILABLE=True` 并提供  
> `load()` 方法自行构建 MNE `RawArray`。  
> **参考文档**: EEGUnity kernel tutorial — "HANDLES_UNAVAILABLE Interface"  
> **规范文档**: `kernels/KERNEL_RULEBOOK.md`

---

## 你的角色

你是一名 EEGUnity kernel 工程师。目标：为给定数据集生成符合 `KERNEL_RULEBOOK.md` 的 Stage1 kernel。  
此 SKILL 适用于 **EEGUnity 会报错无法解析的文件格式**，需在 kernel 内部自行读取数据。  
**在写任何代码前，先完成 Phase 1 全部探索步骤。**

---

## Phase 1: 数据集探索（自动提取）

### 1.1 确认文件格式

```bash
# 列出所有数据文件扩展名
find <dataset_path> -type f | grep -v "__pycache__" | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -20
```

记录：文件扩展名，文件命名规律。

### 1.2 尝试标准 MNE 解析（确认确实无法解析）

```python
import mne
# 根据扩展名尝试对应读取函数：
# .eeg/.vhdr: mne.io.read_raw_brainvision
# .cnt: mne.io.read_raw_cnt
# .set: mne.io.read_raw_eeglab
# .fif: mne.io.read_raw_fif
# 其他: mne.io.read_raw（让 MNE 自动检测）
try:
    raw = mne.io.read_raw("<sample_file>", preload=False)
    print("MNE 可以解析！改用 SKILL_gdf_annotation_remap.md 或其他 SKILL。")
except Exception as e:
    print("MNE 无法解析:", e)
    print("→ 需要 HANDLES_UNAVAILABLE kernel")
```

### 1.3 检查文件结构

针对常见私有格式：

```python
# 若为 .mat 格式：
import scipy.io as sio
mat = sio.loadmat("<sample_file>", simplify_cells=True)
print("keys:", [k for k in mat.keys() if not k.startswith('__')])

# 若为 HDF5 / .h5 格式：
import h5py
with h5py.File("<sample_file>", "r") as f:
    print("keys:", list(f.keys()))
    # 递归展示结构
    f.visititems(lambda name, obj: print(name, type(obj)))

# 若为 numpy / .npy / .npz 格式：
import numpy as np
data = np.load("<sample_file>", allow_pickle=True)
print("shape:", data.shape if hasattr(data, 'shape') else data.files)

# 若为纯二进制格式，查阅官方文档或 README 获取解析方式。
```

记录：数据字段路径、EEG 矩阵的 shape（n_channels × n_samples 还是 n_samples × n_channels）、采样率字段名、通道名字段名、标记/事件字段名。

### 1.4 提取通道信息

```python
# 获取通道名
ch_names = mat["<path>"]["chnames"]   # 根据 1.3 结果替换
print("通道名:", ch_names)
print("数量:", len(ch_names))

# 获取采样率
sfreq = float(mat["<path>"]["sampFreq"])   # 根据 1.3 结果替换
print("采样率:", sfreq, "Hz")
```

识别哪些通道是 EEG、EOG、其他类型。

### 1.5 提取事件 / 标记

参见 `SKILL_mat_sidecar.md` Phase 1.3：定位 marker 字段，判断连续标记还是事件索引。

### 1.6 提取受试者信息

按优先级尝试：

1. `participants.tsv`（BIDS）
2. README / 说明文档
3. 官方论文（DOI / URL）
4. 均无法获取时，所有字段写 `"unknown"`

---

## Phase 2: Kernel 生成

`HANDLES_UNAVAILABLE` kernel 必须实现两个方法：`load()` 和 `apply()`。  
EEGUnity 在检测到文件无法解析时，自动调用 `load()` 代替标准读取流程。

```python
"""<DatasetName> stage1 kernel for EEGUnity (HANDLES_UNAVAILABLE).

<简短描述：数据集名称、范式、文件格式、通道数、事件结构>

This kernel reads <format> files directly because EEGUnity cannot parse
them with standard MNE routines.

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
import scipy.io as sio   # 根据实际格式替换


# Source: <论文引用 / 数据集文档 URL>
# Annotator: <claude-model-id>, <YYYY-MM-DD> — LLM 推断，需人类复核
_SUBJECT_DICT: Dict[str, Dict[str, Any]] = {
    # "<sub_key>": {"age": "<age_str>", "sex": "<sex_str>"},
}

# Source: <数据集文档，说明各 marker 值含义>
# Annotator: <claude-model-id>, <YYYY-MM-DD> — LLM 推断，需人类复核
_EVENT_MAP: Dict[int, str] = {
    # marker_int_value: "annotation_description",
}


def _load_raw_from_file(file_path: str) -> mne.io.BaseRaw:
    """Parse the proprietary file and return an MNE RawArray.

    Adjust the data access paths based on Phase 1.3 findings.
    """
    # 读取文件（根据 Phase 1.3 结果选择）
    mat = sio.loadmat(file_path, simplify_cells=True)
    data_struct = mat["<top_key>"]

    # EEG 数据矩阵
    eeg_data = np.array(data_struct["<data_key>"], dtype=float)
    # 确保 shape 为 (n_channels, n_samples)
    if eeg_data.shape[0] > eeg_data.shape[1]:
        eeg_data = eeg_data.T

    sfreq = float(data_struct["<sfreq_key>"])

    # 通道信息
    # 根据 Phase 1.4 填写实际通道名
    ch_names: List[str] = list(data_struct["<chnames_key>"])
    # 设置通道类型（根据 Phase 1.4 识别结果）
    ch_types = []
    for ch in ch_names:
        if ch.upper().startswith("EOG"):
            ch_types.append("eog")
        elif ch.upper() in ("STI 014", "STIM"):
            ch_types.append("stim")
        else:
            ch_types.append("eeg")

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info, verbose=False)

    # 标准 montage（若通道名符合 standard_1020）
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.info.set_montage(montage, on_missing="ignore")

    # 从 marker 字段构建 annotations（参考 SKILL_mat_sidecar.md 的辅助函数）
    marker = data_struct["<marker_key>"].flatten()
    onsets, durations, descs = [], [], []
    current_code: Optional[int] = None
    current_start: int = 0
    for i, code_raw in enumerate(marker):
        code = int(code_raw)
        label = _EVENT_MAP.get(code)
        if label is not None:
            if current_code is None:
                current_code = code
                current_start = i
            elif code != current_code:
                onsets.append(current_start / sfreq)
                durations.append((i - current_start) / sfreq)
                descs.append(_EVENT_MAP[current_code])
                current_code = code
                current_start = i
        else:
            if current_code is not None:
                onsets.append(current_start / sfreq)
                durations.append((i - current_start) / sfreq)
                descs.append(_EVENT_MAP[current_code])
                current_code = None
    if current_code is not None:
        onsets.append(current_start / sfreq)
        durations.append((len(marker) - current_start) / sfreq)
        descs.append(_EVENT_MAP[current_code])
    raw.set_annotations(mne.Annotations(onsets, durations, descs))

    return raw


@dataclass
class <DatasetName>Kernel:
    """Stage1 kernel for the <dataset_name> dataset (HANDLES_UNAVAILABLE)."""

    KERNEL_ID: str = "<dataset-id>-stage1"
    HANDLES_UNAVAILABLE: bool = True

    def load(self, file_path: str) -> mne.io.BaseRaw:
        """Called by EEGUnity when the file cannot be parsed by standard MNE."""
        return _load_raw_from_file(file_path)

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]

        # ── Subject ID extraction ────────────────────────────────────────────
        subject_match = re.search(r"<pattern>", os.path.basename(file_path), re.IGNORECASE)
        if subject_match is None:
            raise ValueError(f"Cannot infer subject id from: {file_path!r}")
        subject_key = subject_match.group(1).upper()
        if subject_key not in _SUBJECT_DICT:
            raise ValueError(f"Unknown subject id: {subject_key!r}")

        # ── Write eegunity_description ───────────────────────────────────────
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
| 第5条 | 接口 | `HANDLES_UNAVAILABLE = True` 作为类属性存在 |
| 第5条 | 接口 | `load(self, file_path: str) -> mne.io.BaseRaw` 方法存在 |
| 第5条 | 接口 | `apply(self, udataset, raw, row) -> mne.io.BaseRaw` 签名正确 |
| 第6条 | 通道 | `ch_types` 正确区分 eeg / eog / stim，与实际通道对应 |
| 第6条 | 通道 | `set_montage` 使用 `on_missing="ignore"` |
| 第8条 | 描述 | `raw.info["description"]` 写入合法 JSON，包含 `original_description` + `eegunity_description` |
| 第9条 | 描述 | `eegunity_description` 包含全部 5 个字段：`age`、`sex`、`amplifier`、`cap`、`handedness` |
| 第10条 | age | `age` 为字符串类型（`"22"` 而非整数 `22`） |
| 第11条 | sex | `sex` 为小写 `"male"`/`"female"`/`"unknown"` |
| 第12条 | 标签 | `_EVENT_MAP` 中的描述字符串为人类可读，而非裸数字 |
| 第13条 | raw meta | 所有 annotation 均对应原始 marker 值，未引入语义派生 |
| 第16条 | 外部文件 | `load()` 对找不到字段/键的情况有明确错误信息 |
| 第17条 | 注释 | `_SUBJECT_DICT` 和 `_EVENT_MAP` 均有来源注释 |
| 第3条 | LLM诚信 | LLM 推断的信息全部标注"需人类复核" |
