# SKILL: GDF/BDF/EDF Annotation Remap Kernel

> **适用模式**: 数据文件为 GDF / BDF / EDF 格式，事件标签存储于 MNE Annotations 中，  
> 可能为数字编码（如 `"769"`）或混乱字符串，需统一映射为人类可读描述。  
> **参考示例**: `kernels/bcic_iv_2a.py`, `kernels/bcic_iv_2b.py`  
> **规范文档**: `kernels/KERNEL_RULEBOOK.md`

---

## 你的角色

你是一名 EEGUnity kernel 工程师。目标：为给定数据集生成符合 `KERNEL_RULEBOOK.md` 的 Stage1 kernel。  
**在写任何代码前，先完成 Phase 1 全部探索步骤。**

---

## Phase 1: 数据集探索（自动提取）

### 1.1 列出文件结构

```bash
# 替换 <dataset_path> 为实际路径
find <dataset_path> -name "*.gdf" -o -name "*.bdf" -o -name "*.edf" | head -20
```

记录：文件扩展名，文件命名规律（如 `A01T.gdf`、`sub-01_task-MI_eeg.bdf`）。

### 1.2 读取一个代表性样本文件

```python
import mne
# 根据扩展名选择对应的读取函数：
# GDF: mne.io.read_raw_gdf
# BDF: mne.io.read_raw_bdf
# EDF: mne.io.read_raw_edf
raw = mne.io.read_raw_gdf("<sample_file>", preload=False, verbose=False)
print("ch_names:", raw.ch_names)
print("ch_types:", raw.get_channel_types())
print("sfreq:", raw.info['sfreq'])
print("n_channels:", len(raw.ch_names))
```

记录：所有通道名称和类型。识别 EEG 通道、EOG 通道、STI/misc 通道。

### 1.3 提取原始 Annotation 标签

```python
events, event_id = mne.events_from_annotations(raw, verbose=False)
print("event_id:", event_id)
# event_id 是 {'描述字符串': 数字编码} 的字典
# 若描述为纯数字字符串（如 '769'），说明 GDF 存储的是数字码，需要查阅数据集文档翻译
```

记录：所有 `event_id` 条目。若为纯数字，查阅数据集说明文档获取对应含义。

### 1.4 检查通道命名是否标准

对比 `raw.ch_names` 与 MNE 标准名称（`standard_1020`）：

```python
montage = mne.channels.make_standard_montage("standard_1020")
standard_names = set(montage.ch_names)
non_standard = [ch for ch in raw.ch_names if ch not in standard_names]
print("非标准通道名:", non_standard)
```

若有非标准名（如 `EEG:C3`、`EEG-0`、`EEG-Fz`），查阅数据集文档确认对应的标准 10-20 名称。

### 1.5 检查是否存在 .mat sidecar 文件

```bash
find <dataset_path> -name "*.mat" | head -10
```

若存在同名 `.mat` 文件（如 `A01E.mat` 对应 `A01E.gdf`），参见 `SKILL_mat_sidecar.md`。

### 1.6 提取受试者信息

按优先级尝试：

1. **参与者列表文件**（BIDS 格式）：
   ```bash
   find <dataset_path> -name "participants.tsv" | head -3
   ```
2. **数据集 README / 描述文档**：
   ```bash
   find <dataset_path> -name "README*" -o -name "*.pdf" -o -name "*.txt" | head -5
   cat <readme_path>
   ```
3. **数据集官方论文**（DOI / URL，若用户提供）。
4. 若均无法获取，所有字段写 `"unknown"`。

记录：每个受试者 ID → age（str）、sex（`"male"` / `"female"` / `"unknown"`）、  
设备型号（amplifier）、电极帽型号（cap）、利手（handedness）。

---

## Phase 2: Kernel 生成

根据 Phase 1 的提取结果，填写下面的代码模板。  
**⚠️ 注意：LLM 推断的信息必须在注释中标注"LLM 推断，需人类复核"。**

```python
"""<DatasetName> stage1 kernel for EEGUnity.

<一段简短描述：数据集名称、范式、通道数、事件结构>

Source: <论文 DOI 或数据集 URL>
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import mne
import numpy as np


# Source: <论文引用 / 数据集文档 URL>
# Annotator: <claude-model-id>, <YYYY-MM-DD> — LLM 推断，需人类复核
_SUBJECT_DICT: Dict[str, Dict[str, Any]] = {
    # 格式：subject_id -> {age: str, sex: str}
    # age 必须为字符串（"22" 或 "22-30"），sex 必须为 "male"/"female"/"unknown"
    "<SUB_ID>": {"age": "<age_str>", "sex": "<sex_str>"},
    # ... 按 Phase 1.6 的结果填写所有受试者
}

# Source: <数据集文档或论文，说明通道名称来源>
# Annotator: <claude-model-id>, <YYYY-MM-DD> — LLM 推断，需人类复核
# 仅在存在非标准通道名时添加此字典；若已是标准名称则删除此部分
_CHANNEL_RENAME_MAP: Dict[str, str] = {
    # "<原始名>": "<标准 10-20 名>",
    # 例: "EEG:C3": "C3", "EEG-0": "FC3"
}


def _remap_annotations(
    original_events: np.ndarray,
    original_event_id: Dict[str, int],
    new_event_id: Dict[str, int],
    numeric_map: Dict[str, str] = None,
) -> np.ndarray:
    """Map original annotation codes to canonical event IDs.

    numeric_map: 若 GDF 存储数字字符串（如 '769'），提供从数字字符串到
    canonical 描述的翻译表。若 GDF 直接使用描述字符串，传 None。
    """
    remapped = original_events.copy()
    old_to_new: Dict[int, int] = {}

    for orig_desc, orig_id in original_event_id.items():
        canonical = (numeric_map or {}).get(orig_desc, orig_desc)
        new_id = new_event_id.get(canonical)
        if new_id is not None:
            old_to_new[orig_id] = new_id

    for old_id, new_id in old_to_new.items():
        remapped[remapped[:, 2] == old_id, 2] = new_id

    return remapped


@dataclass
class <DatasetName>Kernel:
    """Stage1 kernel for the <dataset_name> dataset."""

    KERNEL_ID: str = "<dataset-id>-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]

        # ── Subject ID extraction ────────────────────────────────────────────
        # 根据 Phase 1.1 的文件命名规律调整提取逻辑
        subject_id = os.path.basename(file_path)[:3]  # 示例：取文件名前3字符
        if subject_id not in _SUBJECT_DICT:
            raise ValueError(f"Unrecognized subject id: {subject_id!r}")

        # ── Step 1: Write eegunity_description ──────────────────────────────
        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "age":        _SUBJECT_DICT[subject_id]["age"],
                "sex":        _SUBJECT_DICT[subject_id]["sex"],
                "amplifier":  "<amplifier_str>",   # 从 Phase 1.6 填写；未知写 "unknown"
                "cap":        "<cap_str>",          # 从 Phase 1.6 填写；未知写 "unknown"
                "handedness": "<handedness_str>",   # "right"/"left"/"unknown"
            },
        }
        raw.info["description"] = json.dumps(description_dict)

        # ── Step 2: Rename channels (仅在存在非标准名时) ────────────────────
        if _CHANNEL_RENAME_MAP:
            rename_map = {
                ch: _CHANNEL_RENAME_MAP[ch]
                for ch in raw.ch_names
                if ch in _CHANNEL_RENAME_MAP
            }
            if rename_map:
                raw.rename_channels(rename_map)

        # ── Step 3: Set channel types (EOG / STI) ──────────────────────────
        # 根据 Phase 1.2 识别的 EOG 通道，更新类型
        # 若所有通道类型已正确，跳过此步
        # 示例（按需修改通道名匹配逻辑）：
        # _EOG_KEYS = {"EOG:CH01", "EOG:CH02"}  # 用实际名称替换
        # ch_type_updates = {ch: "eog" for ch in raw.ch_names if ch.upper() in _EOG_KEYS}
        # if ch_type_updates:
        #     raw.set_channel_types(ch_type_updates)

        # ── Step 4: Set standard montage ────────────────────────────────────
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.info.set_montage(montage, on_missing="ignore")

        # ── Step 5: Remap events ─────────────────────────────────────────────
        # 根据 Phase 1.3 填写 canonical event_id 和 numeric_map
        event_id = {
            # "<描述字符串>": <整数 ID>,
            # 例: "Cue onset left (class 1)": 7,
        }

        # 若 GDF annotation 为纯数字字符串（如 '769'），填写此翻译表
        # 若 annotation 已是描述字符串，设为 None
        numeric_map = {
            # "<数字字符串>": "<canonical 描述>",
            # 例: "769": "Cue onset left (class 1)",
        } or None  # 删除此行并直接赋 None 若不需要

        original_events, original_event_id = mne.events_from_annotations(raw, verbose=False)
        new_events = _remap_annotations(
            original_events, original_event_id, event_id, numeric_map
        )

        # ── Step 6: Convert back to annotations ─────────────────────────────
        id_to_desc = {v: k for k, v in event_id.items()}
        annotations = mne.annotations_from_events(
            events=new_events,
            sfreq=raw.info["sfreq"],
            event_desc=id_to_desc,
        )
        raw.set_annotations(annotations)
        return raw


KERNEL = <DatasetName>Kernel()
```

---

## Phase 3: 验证清单（对照 KERNEL_RULEBOOK.md）

生成代码后逐条核查：

| 编号 | 规则 | 检查项 |
|------|------|--------|
| 第5条 | 接口 | 模块末尾有 `KERNEL = <ClassName>()` |
| 第5条 | 接口 | `apply(self, udataset, raw, row) -> mne.io.BaseRaw` 签名正确 |
| 第6条 | 通道 | channel rename 在 `set_montage()` **之前**调用 |
| 第6条 | 通道 | `set_montage` 使用 `on_missing="ignore"` |
| 第7条 | 来源 | `_CHANNEL_RENAME_MAP` 有来源注释（论文/文档 + 标注者 + 日期） |
| 第8条 | 描述 | `raw.info["description"]` 写入合法 JSON，包含 `original_description` + `eegunity_description` |
| 第9条 | 描述 | `eegunity_description` 包含全部 5 个字段：`age`、`sex`、`amplifier`、`cap`、`handedness` |
| 第10条 | age | `age` 为字符串类型（`"22"` 而非整数 `22`） |
| 第10条 | age | `age` 未知时为 `"unknown"`，不得为 `"0"` 或空字符串 |
| 第11条 | 描述来源 | `_SUBJECT_DICT` 有来源注释；若 LLM 填写则标注"需人类复核" |
| 第11条 | sex | `sex` 为小写 `"male"`/`"female"`/`"unknown"`（不得有大写首字母） |
| 第12条 | 标签 | annotation 描述为人类可读字符串，而非裸数字 |
| 第13条 | raw meta | 所有 annotation 均为 raw meta（直接对应原始事件码） |
| 第17条 | 注释 | 所有非原始数据来源均有：来源 + 标注者 + 日期 的注释 |
| 第3条 | LLM诚信 | LLM 不以人类身份标注；LLM 填写的信息全部标注"需人类复核" |
