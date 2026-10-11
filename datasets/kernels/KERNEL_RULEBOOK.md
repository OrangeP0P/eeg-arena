# EEG Dataset Kernel 编写与维护规则

状态：草案 v0.1

最近确认：2026-04-08

适用对象：kernel 编写者（人类、LLM）、维护者（人类、LLM）

适用范围：`app_kernel/kernels/` 目录下所有 kernel 文件

---

# 第一编 总则

## 第一章 目的与适用范围

### 第1条 目的

本规则用于统一 EEG 数据集 kernel 文件的编写、维护与扩展行为，确保 kernel 文件能够：

1. 与 EEGUnity 现行接口规范保持兼容。
2. 尽可能准确还原原始数据中缺失或编码混乱的通道信息。
3. 为 EEGUnity 提供标准化的被试者与设备元信息。
4. 输出规范化的标签结构，支持 raw meta 与 super meta 的动态实例化。
5. 作为可维护的增强数据容器，对所有非原始来源的信息记录来源、添加日期与责任归属。

### 第2条 适用范围

本规则适用于 `app_kernel/kernels/` 目录下所有 kernel 文件的以下行为：

1. 新建与初始化。
2. 人工扩展或 LLM 辅助扩展。
3. 审核与复核。

### 第3条 基本原则

1. **接口兼容性优先**：kernel 必须与当前版本 EEGUnity / MNE 接口保持兼容，不得引入破坏标准接口的自定义封装。
2. **来源可追溯**：kernel 文件中所有非原始数据来源的信息均须在紧邻的代码注释中标注来源、标注者与日期。
3. **LLM 标记诚信原则**：LLM 不得以人类身份标记任何信息。除非收到人类的明确指令，LLM 添加的所有内容必须注明"由 LLM 推断，需人类复核"。

---

# 第二编 接口规范

## 第二章 EEGUnity Kernel 接口

### 第4条 接口标准参考

kernel 文件必须遵守 EEGUnity 现行的 kernel 接口规范。本规则不重复定义接口细节；维护者应查阅 EEGUnity 相关函数与文档（包括但不限于 `UnifiedDataset.eeg_parser`、`EEGBatch` 及其 kernel 加载机制）以确认最新接口要求。

### 第5条 接口一致性要求

1. kernel 文件必须暴露模块级别的 `KERNEL` 对象，供 EEGUnity 加载与调用。
2. kernel 的核心方法签名与返回值必须与当前版本 EEGUnity 保持一致；若 EEGUnity 升级导致接口变动，kernel 文件须同步更新。
3. kernel 文件内部的辅助函数与数据结构不受接口约束，但不得影响 `KERNEL` 对象的标准行为。

---

# 第三编 通道信息还原

## 第三章 通道信息

### 第6条 通道信息还原义务

1. kernel 应尽可能将原始数据中编码混乱或缺失的通道名称还原为标准名称（如 `EEG-0` → `FC3`）。还原依据应来自论文、数据集说明文档或数据集提供方的官方资料。
2. 通道类型解析依赖 EEGUnity 的 `format_channel_name()` 函数，kernel **不强制要求**对所有通道显式设置 `type`。若 kernel 确实调用 `set_channel_types()`，其结果应与 `format_channel_name()` 的解析结果保持一致，不得矛盾。
3. 若原始数据使用标准帽（如 standard_1020），kernel 应在通道重命名后调用 `set_montage()` 设置空间位置信息。
4. 若原始数据使用非标准帽或无法确认电极布局，可跳过 montage 设置，但须在 kernel 文件顶部注释中说明原因。

### 第7条 非原始来源的通道信息

1. kernel 文件中包含的通道重映射信息，若**非来自原始数据文件本身**（即来自论文、数据集说明、人工标记、LLM 推断等外部来源），必须在对应代码处以注释标注：
   - **来源**：论文（附引用或 DOI）、数据集 README（附路径）、人工指令等。
   - **标注者**：人类审核员（附姓名或 ID）或 LLM（附模型标识）。
   - **日期**：信息添加的日期（ISO 格式，如 `2026-04-08`）。
2. 若由 LLM 填写，必须额外注明"由 LLM 推断，需人类复核"。人类审核员确认后，可移除该标记，并更新标注者为人类。
3. LLM 不得以人类身份标记任何通道信息，除非收到人类的明确书面指令。

---

# 第四编 被试者与设备元信息

## 第四章 eegunity_description 格式

### 第8条 写入路径

被试者与设备元信息必须以 JSON 字符串的形式写入 `raw.info["description"]`，格式如下：

```json
{
  "original_description": "<原始 description 字段值，字符串>",
  "eegunity_description": {
    "age": "<str>",
    "sex": "<str>",
    "amplifier": "<str>",
    "cap": "<str>",
    "handedness": "<str>"
  }
}
```

其中 `original_description` 应保留原始 `raw.info["description"]` 的值（若原始值为空，写空字符串）。

### 第9条 必填字段

`eegunity_description` 中以下字段均为**必填**，不得省略：

| 字段 | 含义 | 合法值 |
|------|------|--------|
| `age` | 被试者年龄 | 正整数字符串（`"22"`）；或范围字符串（`"22-45"`）；未知时写 `"unknown"` |
| `sex` | 被试者性别 | `"female"` 或 `"male"`；未知时写 `"unknown"` |
| `amplifier` | 采集设备型号 | 设备名称字符串（如 `"BrainAmp"`）；未知时写 `"unknown"` |
| `cap` | 电极帽材质或型号 | 材质字符串（如 `"Ag/AgCl"`）；未知时写 `"unknown"` |
| `handedness` | 利手方向 | `"right"`、`"left"`；未知时写 `"unknown"` |

所有字段无论是否已知均**必须出现**。未知值一律写字符串 `"unknown"`，不得写 `null`、空字符串或 Python `None`。

### 第10条 age 字段约定

1. `age` 统一以字符串（`str`）存储，不得使用整数或浮点数类型。
2. 若已知精确年龄，写正整数字符串，如 `"22"`。
3. 若仅知年龄范围（如论文描述"22–45岁"），写范围字符串，格式为 `"<下界>-<上界>"`，如 `"22-45"`；两端均为正整数，中间用英文连字符 `-` 连接。
4. 未知时写 `"unknown"`，不得写 `"0"` 或负数字符串。

### 第11条 被试者信息的来源标注

1. 若 `eegunity_description` 中任何字段的值来自外部资料（论文、数据集说明文档、人工录入），必须在 kernel 文件中对应的 subject dict 或赋值代码处以注释说明来源、标注者与日期。
2. 若由 LLM 填写，所有字段值须设置为 `"unknown"`，并注明"由 LLM 推断，需人类复核"，除非收到人类的明确指令提供具体值。
3. LLM 不得在未获人类指令的情况下在 subject dict 中填写具体的 age、sex 等实质性数值。

---

# 第五编 标签标准化

## 第五章 标签形式与职责

### 第12条 kernel 的标签职责

kernel 负责将原始数据中的标签信息标准化，输出为 EEGUnity / MNE 可稳定解析的格式。具体职责包括：

1. 将原始数字编码的 annotation 映射为人类可读的描述字符串。
2. 确保输出标签通过 `mne.events_from_annotations()` 可被稳定解析。
3. 若数据集的标签来源于 STIM channel 或 MISC channel，kernel 可选择将其转化为 annotation 形式输出，或保留原始通道形式，但须在 kernel 文件中以注释说明选择依据。

### 第13条 raw meta

1. **raw meta**（对应 `analysis.md` 中 `Meta Level = raw`）：kernel 输出的标签直接对应原始数据中的事件编码，经过名称标准化但不引入任何语义派生。
2. raw meta 是所有 kernel **必须实现**的基础层，不得缺失。
3. raw meta 事件的描述字符串应与数据集 `analysis.md` 的 `### Event Structure` 表格中的 `Event ID` 列保持一致；若二者不一致，应以 kernel 实际输出为准并同步修正 `analysis.md`。

### 第13条之1 多范式数据集的范式前缀约定

当单一数据集包含**多个范式**（paradigm），且不同范式之间存在**重叠的事件编码**（即相同数字码在不同范式中代表不同语义），kernel 应在所有事件描述字符串前加上范式标识前缀，以消除歧义。

1. **触发条件**：同一数据集下存在多个范式，且不同范式的事件编码存在重叠（如 CLA 和 HaLT 均使用编码 1 表示左手 MI，但 NoMT 使用编码 1 表示左手 cue）。
2. **前缀格式**：`<PARADIGM>: <event_label>`，其中 `<PARADIGM>` 为范式名称（大写或数据集约定的缩写），`<event_label>` 为该范式中该编码的语义描述。示例：`CLA: left hand MI`、`HaLT: right hand MI`、`NoMT: left hand cue`。
3. **分类依据**：此类标签仍归类为 **raw meta**，因为范式标识来源于原始文件名或路径前缀，不引入任何语义推断；前缀提取本质上是文件名解析，非语义派生。
4. **行政标记（admin codes）**：若数据集中存在用途不明确的管理标记编码（如 91 / 92 / 99），在尚未确认其具体含义的情况下，统一使用 `admin: marker-<code>` 格式，**不加范式前缀**，以标示其跨范式的共用属性，并在 kernel 文件顶部注释中说明已知的观测规律。
5. **同步要求**：范式前缀的引入必须同步更新对应数据集的 `analysis.md` 中 `### Event Structure` 表格，确保 `Event ID` 列与 kernel 实际输出保持一致。
6. **范式检测逻辑**：kernel 文件应提供独立的范式检测函数（如 `_detect_paradigm(file_name)`），基于文件名前缀进行匹配，并在 kernel 文件顶部注释中说明检测顺序（如 FREEFORM 须先于 FREE 或 CLA 检查，以防前缀歧义）。

```python
# 示例：范式前缀事件映射表
_PARADIGM_LABEL_MAP = {
    "CLA": {1: "CLA: left hand MI", 2: "CLA: right hand MI"},
    "HaLT": {1: "HaLT: left hand MI", 2: "HaLT: right hand MI",
             91: "admin: marker-91"},  # admin codes: no paradigm prefix
    "NoMT": {1: "NoMT: left hand cue"},  # passive cue, not MI
}
```

### 第14条 super meta

1. **super meta**（对应 `analysis.md` 中 `Meta Level = super`）：kernel 可在 raw meta 基础上进一步构建语义派生标签，例如将多个 raw 事件组合为高层试次类型，或引入外部信息对事件打标。
2. super meta 与 raw meta 的存储形式相同（均写入 MNE Raw 的 annotations 或相应通道中）。
3. super meta 的引入**必须**与数据集 `analysis.md` 中 `Tasks Area` 对应任务的 `Meta Level` 字段一致（参见主规则书 `app/rulebook.md` 相关条款）。kernel 不得单方面引入在 `analysis.md` 中未声明的 super meta 标签。
4. LLM 可根据 `analysis.md` 的内容推断并构建 super meta，但须在 kernel 文件中明确标注"由 LLM 构建"并说明推断依据；人类审核员确认后方可移除"需人类复核"标记。
5. super meta 的构建逻辑应在 kernel 文件中以独立的数据结构或注释块清晰标注，不得与 raw meta 逻辑混写且无注释区分。

---

# 第六编 静态增强数据与外部资源

## 第六章 kernel 文件中的增强数据

### 第15条 允许在 kernel 文件中包含的非原始数据

kernel 文件允许包含以下类型的静态增强数据：

| 数据类型 | 典型示例 | 推荐命名风格 |
|----------|----------|--------------|
| 被试者信息字典 | `_SUBJECT_DICT` | 下划线前缀私有变量 |
| 通道重映射表 | `_CHANNEL_RENAME_MAP`、`alias_groups` | 下划线前缀私有变量 |
| 事件编码映射表 | `_NUMERIC_EVENT_MAP` | 下划线前缀私有变量 |
| super meta 构建规则 | `_SUPER_META_RULES` | 下划线前缀私有变量 |
| 设备 / 帽信息 | 硬编码字符串常量 | 大写常量或 subject dict 内嵌 |

以上数据均视为 kernel 文件的合法组成部分。数据的存在本身意味着它已经过（或待经过）人类或 LLM 的审核，必须附有来源注释（见第16条）。

### 第16条 外部文件读取

1. kernel 文件允许读取外部文件（如同目录或指定路径下的 `.json`、`.csv`、`.mat` 等）以补充或替代硬编码数据。
2. 读取外部文件时，须在 kernel 文件中以注释说明外部文件的用途、格式与来源。
3. 外部文件的读取逻辑不得破坏 kernel 的接口兼容性：即使外部文件缺失，kernel 也必须有合理的降级处理（如回退到内置默认值，或抛出包含明确说明的错误信息）。

### 第17条 来源注释规范

对于 kernel 文件中所有非原始数据来源的信息，必须在**紧邻的代码注释**中说明以下内容：

1. **来源**：论文（附引用或 DOI）、数据集说明文档（附路径或 URL）、人工录入、LLM 推断等。
2. **标注者**：人类审核员（附姓名或缩写）或 LLM（附模型标识，如 `claude-sonnet-4-6`）。
3. **日期**：信息添加或最近确认的日期（ISO 格式，如 `2026-04-08`）。
4. LLM 标注时，必须额外注明"需人类复核"；人类审核员确认后，由人类将标注者更新为自己并移除"需人类复核"标记。

示例注释格式（Python）：

```python
# Source: BCI Competition IV Dataset 2a description, https://www.bbci.de/competition/iv/
# Annotator: [human-reviewer-id], 2026-04-08
_SUBJECT_DICT = { ... }

# Source: Inferred from channel naming convention in dataset README
# Annotator: claude-sonnet-4-6, 2026-04-08 — 需人类复核
_CHANNEL_RENAME_MAP = { ... }
```
