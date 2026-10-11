# SKILL: BIDS Standard Dataset Kernel

> **适用模式**: 数据集遵循 BIDS 标准，包含 `participants.tsv` 和 `*_events.tsv`，  
> EEGUnity 可正常读取 EEG 文件（EDF / BDF / SET 等）。  
> kernel 的主要任务是从 BIDS sidecar 文件中提取被试者元信息和事件结构。  
> **参考示例**: `kernels/bids.py`（通用 BIDS kernel，可直接复用）  
> **规范文档**: `kernels/KERNEL_RULEBOOK.md`

---

## 你的角色

你是一名 EEGUnity kernel 工程师。目标：为给定 BIDS 数据集生成符合 `KERNEL_RULEBOOK.md` 的 Stage1 kernel。  
**首先确认通用 BIDS kernel 是否已经满足需求；仅在需要数据集特定处理时才新建 kernel。**

---

## Phase 0: 是否需要新建 kernel？

首先尝试使用通用 BIDS kernel（`kernels/bids.py`）：

```python
# app_kernel/kernels/bids.py 已实现：
# - 从 participants.tsv 提取 age / sex
# - 从 *_events.tsv 或 raw.annotations 提取事件摘要
# - 写入 eegunity_description（不包含 amplifier/cap/handedness）
```

只有在以下情况下才需要新建数据集专属 kernel：
1. 通道名称不标准，需要重命名
2. 事件需要重映射（events.tsv 的标签格式特殊）
3. 需要读取额外的 sidecar 文件（如 `.mat`、`.json`）
4. amplifier / cap / handedness 有已知值需要硬编码
5. 存在多范式文件需要范式前缀

若通用 kernel 已满足，直接在 analysis.md 中指定 `bids.py` 即可，无需新建文件。

---

## Phase 1: 数据集探索（自动提取）

### 1.1 确认 BIDS 结构

```bash
# 检查顶层结构
ls <dataset_path>/
# 期望：participants.tsv, dataset_description.json, sub-*/ 目录

# 检查受试者目录结构
ls <dataset_path>/sub-01/
# 期望：eeg/ 目录（或 ses-XX/eeg/）
```

### 1.2 读取 participants.tsv

```python
import pandas as pd
df = pd.read_csv("<dataset_path>/participants.tsv", sep="\t")
print(df.head(10))
print("列名:", df.columns.tolist())
print("行数:", len(df))
```

记录：`participant_id` 列的格式，`age` 列（是否存在，格式），`sex`/`gender` 列（值的格式）。

### 1.3 读取一个样本 EEG 文件

```bash
find <dataset_path> -name "*.edf" -o -name "*.bdf" -o -name "*.set" | head -3
```

```python
import mne
raw = mne.io.read_raw("<sample_eeg_file>", preload=False, verbose=False)
print("ch_names:", raw.ch_names)
print("ch_types:", raw.get_channel_types())
print("sfreq:", raw.info['sfreq'])
```

记录：是否存在非标准通道名。

### 1.4 读取 events.tsv

```bash
find <dataset_path> -name "*_events.tsv" | head -5
```

```python
events_df = pd.read_csv("<sample_events_tsv>", sep="\t")
print(events_df.head(10))
print("列名:", events_df.columns.tolist())
# 标准列：onset, duration, trial_type, value, stim_type
```

记录：事件标签列名，所有唯一标签值及其含义。

### 1.5 检查 dataset_description.json

```python
import json
with open("<dataset_path>/dataset_description.json") as f:
    desc = json.load(f)
print(json.dumps(desc, indent=2))
# 查找: Name, Authors, ReferencesAndLinks, EEGReference 等
```

记录：数据集名称、引用论文 DOI，设备信息（若有）。

### 1.6 确认通用 bids.py 是否可用

| 检查项 | 结果 |
|--------|------|
| participants.tsv 存在且有 age / sex 列 | ✓ / ✗ |
| 通道名符合 standard_1020 | ✓ / ✗ |
| events.tsv 标签已是可读字符串 | ✓ / ✗ |
| amplifier / cap 有已知硬编码值 | ✓ / ✗ |
| 无多范式歧义 | ✓ / ✗ |

若全为 ✓，使用通用 `bids.py`，本 SKILL 流程结束。  
若有 ✗，继续 Phase 2 新建专属 kernel。

---

## Phase 2: 专属 Kernel 生成

仅在通用 `bids.py` 不满足时使用此模板。  
可从 `bids.py` 的 `BIDSKernel` 继承，只覆盖需要扩展的部分。

```python
"""<DatasetName> stage1 kernel for EEGUnity.

Extends the generic BIDS kernel with dataset-specific channel mapping
and/or hardcoded device metadata.

Source: <数据集 DOI 或 OpenNeuro 链接>
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import mne

# 继承通用 BIDS kernel（已处理 participants.tsv / events.tsv）
from kernels.bids import BIDSKernel


# Source: <数据集文档>
# Annotator: <claude-model-id>, <YYYY-MM-DD> — LLM 推断，需人类复核
# 仅在通道名不标准时添加此字典
_CHANNEL_RENAME_MAP: Dict[str, str] = {
    # "<原始名>": "<标准 10-20 名>",
}

# 已知的设备元信息（若数据集文档明确记载）
# Source: <数据集文档 / 论文>
# Annotator: <claude-model-id>, <YYYY-MM-DD> — LLM 推断，需人类复核
_AMPLIFIER = "<amplifier_str>"   # 未知写 "unknown"
_CAP = "<cap_str>"               # 未知写 "unknown"
_HANDEDNESS = "<handedness_str>" # 未知写 "unknown"


@dataclass
class <DatasetName>Kernel(BIDSKernel):
    """Stage1 kernel for the <dataset_name> BIDS dataset."""

    KERNEL_ID: str = "<dataset-id>-stage1"

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        file_path = row["File Path"]

        # ── Step 1: Rename channels (仅在存在非标准名时) ────────────────────
        if _CHANNEL_RENAME_MAP:
            rename_map = {
                ch: _CHANNEL_RENAME_MAP[ch]
                for ch in raw.ch_names
                if ch in _CHANNEL_RENAME_MAP
            }
            if rename_map:
                raw.rename_channels(rename_map)

        # ── Step 2: Set montage ──────────────────────────────────────────────
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.info.set_montage(montage, on_missing="ignore")

        # ── Step 3: Get age / sex from participants.tsv (via parent class) ──
        age, sex = self.get_age_sex(file_path)

        # ── Step 4: Write eegunity_description ──────────────────────────────
        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "age":        age,
                "sex":        sex,
                "amplifier":  _AMPLIFIER,
                "cap":        _CAP,
                "handedness": _HANDEDNESS,
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
| 第6条 | 通道 | channel rename 在 `set_montage()` **之前**调用 |
| 第6条 | 通道 | `set_montage` 使用 `on_missing="ignore"` |
| 第7条 | 来源 | `_CHANNEL_RENAME_MAP` 有来源注释（若存在） |
| 第8条 | 描述 | `raw.info["description"]` 写入合法 JSON，包含 `original_description` + `eegunity_description` |
| 第9条 | 描述 | `eegunity_description` 包含全部 5 个字段：`age`、`sex`、`amplifier`、`cap`、`handedness` |
| 第10条 | age | `age` 为字符串；未知时为 `"unknown"` 而非 `"0"` |
| 第11条 | sex | `sex` 为小写 `"male"`/`"female"`/`"unknown"` |
| 第13条 | raw meta | 若修改了 annotations，确认仍为 raw meta（无语义派生） |
| 第17条 | 注释 | `_CHANNEL_RENAME_MAP`、`_AMPLIFIER`/`_CAP` 有来源注释 |
| 第3条 | LLM诚信 | LLM 推断的信息全部标注"需人类复核" |

---

## 附：通用 bids.py 的局限性说明

通用 `bids.py` 不填写 `amplifier`、`cap`、`handedness` 字段（写入 `"unknown"`），  
且不做通道重命名。如果数据集有这些已知信息，必须新建专属 kernel 并硬编码。
