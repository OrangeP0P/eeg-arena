<h1 align="center">EEG-Arena：大规模 EEG 模型评测平台</h1>

<p align="center"><strong>中文</strong> | <a href="README.md">English</a></p>

<p align="center"><em>Unified data interface and experiment infrastructure for large-scale EEG foundation models</em></p>

<p align="center">
  <a href="docs/model_interface.md"><img alt="MODEL_INTERFACE" src="https://img.shields.io/badge/MODEL__INTERFACE-F3F5F7?style=flat-square&logo=gitbook&logoColor=555555&labelColor=F3F5F7&color=F3F5F7"></a>
  <a href="https://eeg.fm/leaderboards/"><img alt="LEADERBOARD" src="https://img.shields.io/badge/LEADERBOARD-F3F5F7?style=flat-square&logo=starship&logoColor=555555&labelColor=F3F5F7&color=F3F5F7"></a>
  <a href="https://eegarena.eeg.fm/"><img alt="BENCHMARK_RESULTS" src="https://img.shields.io/badge/BENCHMARK__RESULTS-F3F5F7?style=flat-square&logo=plotly&logoColor=555555&labelColor=F3F5F7&color=F3F5F7"></a>
</p>

<p align="center">
  <a href="#-%E6%A8%A1%E5%9E%8B%E5%BA%93"><img alt="30 EEG FMs" src="https://img.shields.io/badge/EEG%20FMs-30-0A7EA4?style=flat-square"></a>
  <a href="#-%E6%A8%A1%E5%9E%8B%E5%BA%93"><img alt="25 传统模型" src="https://img.shields.io/badge/%E4%BC%A0%E7%BB%9F%E6%A8%A1%E5%9E%8B-25-5B8C5A?style=flat-square"></a>
  <a href="datasets/__init__.py"><img alt="29 数据集 / 74 任务" src="https://img.shields.io/badge/%E6%95%B0%E6%8D%AE%E9%9B%86%20%2F%20%E4%BB%BB%E5%8A%A1-29%20%2F%2074-2D6A4F?style=flat-square"></a>
  <a href="requirements.txt"><img alt="PyTorch 2.5.1" src="https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="#%EF%B8%8F-%E9%83%A8%E7%BD%B2%E5%B7%A5%E5%85%B7"><img alt="单任务与批量工作流" src="https://img.shields.io/badge/%E5%8D%95%E4%BB%BB%E5%8A%A1%20%2B%20%E6%89%B9%E9%87%8F%E5%B7%A5%E4%BD%9C%E6%B5%81-6C757D?style=flat-square"></a>
</p>

---

## 🚩 News

- **[2026-10-10]** 我们开源了 EEG-Arena 评测工具包 v0.00，并撰写了使用教程。
- **[2026-09-30]** 我们建立了公开的 [模型排行榜](https://eeg.fm/) 与 [评测结果网站](https://eegarena.eeg.fm/)。
- **[2026-09-30]** [EEG-Arena 完整论文](https://arxiv.org/abs/2609.32743) 已上线。
- **[2026-09-25]** EEG-Arena 所使用的数据工具 [NeuroDoc](https://neurodoc.eeg.fm/) 已被 NeurIPS 2026 接收。

## 🧠 项目简介

`EEG-Arena` 是一个面向大规模 EEG 基础模型的统一数据与实验基础设施，支持预训练模型和传统监督基线模型的统一接入、运行，以及大规模实验的公平评测、调度与复现。

## ✨ 项目亮点

- **数据标准化**：面向大规模异质脑电数据的统一规范与数据接口。
- **模型标准化**：针对 EEG 模型的统一代码规范与接入接口。
- **统一评测**：支持大规模实验部署、调度与结果汇总。
- **公开结果**：[Leaderboard](https://eeg.fm/leaderboards/) · [Evaluation Results](https://eegarena.eeg.fm/)。
- **数据工具**：通过 [NeuroDoc](https://neurodoc.eeg.fm/) 持续、大规模扩展下游数据集。

## 🚀 Quick Start

### Step 1：🛠️ 安装所需环境

克隆仓库后，创建名为 `eeg-arena` 的 Conda 环境，并使用仓库提供的 [requirements.txt](requirements.txt) 安装依赖：

```bash
conda create -n eeg-arena python=3.12 -y
conda activate eeg-arena
pip install -r requirements.txt
```

EEGMamba 还需额外安装 CUDA 扩展。请先确保已安装与 PyTorch 匹配的 CUDA toolkit、Git 和 GCC 9+，再执行：

```bash
pip install ninja
CAUSAL_CONV1D_FORCE_BUILD=TRUE \
pip install "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0" \
  --no-build-isolation
MAMBA_FORCE_BUILD=TRUE \
pip install "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@v2.2.2" \
  --no-build-isolation
```

### Step 2：📦 下载和准备预训练模型权重

将预训练权重统一放在 `models/Weights/` 下，并保留各模型配置所要求的目录名与文件名。

> [!IMPORTANT]
> 评测所需权重合计约 `10.2 GiB`（`11.0 GB`），建议至少预留 `12 GB` 磁盘空间。完整目录结构如下：

<details>
<summary><img alt="📂 点击展开完整权重目录结构" src="https://img.shields.io/badge/%F0%9F%93%82%20%E7%82%B9%E5%87%BB%E5%B1%95%E5%BC%80-%E5%AE%8C%E6%95%B4%E6%9D%83%E9%87%8D%E7%9B%AE%E5%BD%95-0A7EA4?style=flat-square"></summary>

```text
models/
└── Weights/
    ├── BENDR/
    │   ├── contextualizer.pt
    │   └── encoder.pt
    ├── BIOT/
    │   ├── EEG-PREST-16-channels.ckpt
    │   ├── EEG-SHHS+PREST-18-channels.ckpt
    │   └── EEG-six-datasets-18-channels.ckpt
    ├── EEGPT/
    │   └── eegpt_mcae_58chs_4s_large4E.ckpt
    ├── LaBraM/
    │   └── labram-base.pth
    ├── NeuroGPT/
    │   └── pytorch_model.bin
    ├── CBraMod/
    │   └── pretrained_weights.pth
    ├── CSBrain/
    │   └── CSBrain.pth
    ├── EEGMamba/
    │   └── pretrained_EEGMamba.pth
    ├── HEAR/
    │   ├── base/checkpoint.pth
    │   └── large/checkpoint-49.pth
    ├── LEAD/
    │   └── p-base/checkpoint.pth
    ├── LUNA/
    │   ├── LUNA_base.safetensors
    │   ├── LUNA_large.safetensors
    │   └── LUNA_huge.safetensors
    ├── mdJPT/
    │   └── mdjpt_pretrained.pth
    ├── NeuroRVQ/
    │   └── NeuroRVQ_EEG_foundation_model_v1.pt
    ├── REVE/
    │   ├── reve-base.safetensors
    │   └── reve-large.safetensors
    ├── CodeBrain/
    │   └── CodeBrain.pth
    ├── STEEGFormer/
    │   ├── small/checkpoint-300.pth
    │   ├── base/checkpoint-288.pth
    │   └── large/large_weights_only_196.pth
    ├── SleepFM/
    │   ├── model_sleep_staging_best.pth
    │   ├── model_diagnosis_best.pth
    │   └── model_base_best.pt
    ├── SleepGPT/
    │   └── ModelCheckpoint-epoch=79-val_acc=0.0000-val_score=4.2305.ckpt
    ├── TFMTokenizer/
    │   └── github/single_dataset_settings/CHBMIT_tfm_tokenizer_2x2x8/
    │       ├── tfm_encoder_best_model.pth
    │       └── tfm_tokenizer_last.pth
    └── UniDMFM/
        └── pretrained_weights.pth
```

</details>

请从下表中的官方来源下载权重，并按照上方目录结构放置：

| 模型 | 权重下载 |  | 模型 | 权重下载 |
|---|---|:---:|---|---|
| BENDR | [GitHub Release](https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha) | │ | BIOT | [GitHub](https://github.com/ycq091044/BIOT/tree/main/pretrained-models) |
| EEGPT | [Figshare](https://figshare.com/s/e37df4f8a907a866df4b) | │ | LaBraM | [GitHub](https://github.com/935963004/LaBraM/tree/main/checkpoints) |
| NeuroGPT | [Hugging Face](https://huggingface.co/wenhuic/Neuro-GPT) | │ | CBraMod | [Hugging Face](https://huggingface.co/weighting666/CBraMod) |
| CSBrain | [Google Drive](https://drive.google.com/drive/folders/1-GsVVewRM0B93H08yts5m53yU2whxYvj?usp=sharing) | │ | EEGMamba | [Hugging Face](https://huggingface.co/weighting666/EEGMamba) |
| HEAR | [GitHub](https://github.com/OrangeP0P/HEAR) | │ | LEAD | [Google Drive](https://drive.google.com/drive/folders/1_XUfU3vZB40rjivkNYf8L2slCahXPo43) |
| LUNA | [Hugging Face](https://huggingface.co/PulpBio/LUNA) | │ | NeuroRVQ | [Hugging Face](https://huggingface.co/ntinosbarmpas/NeuroRVQ) |
| REVE | [Base](https://huggingface.co/brain-bzh/reve-base) · [Large](https://huggingface.co/brain-bzh/reve-large) | │ | CodeBrain | [Hugging Face](https://huggingface.co/YjMajy/CodeBrain) |
| ST-EEGFormer | [GitHub Releases](https://github.com/LiuyinYang1101/STEEGFormer/releases) | │ | SleepFM | [GitHub](https://github.com/zou-group/sleepfm-clinical/tree/main/sleepfm/checkpoints) |
| SleepGPT | [Figshare](https://doi.org/10.6084/m9.figshare.30626870) | │ | TFM-Tokenizer | [GitHub](https://github.com/Jathurshan0330/TFM-Tokenizer/tree/main/pretrained_weigths) |
| Uni-DMFM | [Google Drive](https://drive.google.com/file/d/1Dlk6X3YYmJfOjNKu7xaehzLFAIgnATIM/view?usp=drive_link) | │ |  |  |

### Step 3：🗂️ 准备评测数据集

EEG-Arena 支持以下两种数据集文件准备方式，如果你想快速开始，使用 ⚡ 第一种方式：

> [!IMPORTANT]
> 全量评测需预留至少 **3 TB** 存储空间，用于存放全部评测数据缓存。

| 数据准备方式 | 适用场景 | 下载链接 | 说明 |
|---|---|---|---|
| ⚡ 下载标准化数据 | 快速复现实验，直接使用我们提供的标准化下游数据文件 | [ModelScope](https://www.modelscope.cn/datasets/orangeeg/eeg-arena) | [数据集说明](https://eeg.fm/benchmarks/datasets/) |
| 🛠️ 从原始数据构建 | 使用 EEG-Arena 数据处理脚本，将本地原始数据集导出为下游格式 | [下载索引](https://eeg.fm/datasets/) | 框架按需构建并复用 HDF5 缓存 |

#### 方式一：⚡ 使用标准化数据文件

完成标准化数据文件下载后，将其放入 `./data/cache`，或在运行配置中通过 `cache_dir` 指定其他路径，无需额外准备原始数据。

解压后的缓存目录结构如下：

```text
data/
└── cache/
    └── <dataset>/
        ├── locator.csv
        └── <task>/
            └── <config_key>.hdf5
```

#### 方式二：🛠️ 从原始数据构建缓存

从原始数据构建缓存时，每个数据集应单独放在一个子目录中。例如，运行参数为 `--dataset bcic_iv_2b` 时，程序会从 `data/raw/bcic_iv_2b/` 中读取原始文件：

```text
data/
└── raw/
    ├── bcic_iv_2b/
    │   ├── B0101T.gdf
    │   ├── B0102T.gdf
    │   ├── B0104E.gdf
    │   ├── B0104E.mat
    │   └── ...
    ├── physionet_eegmmidb/
    │   └── ...
    └── <dataset>/
        └── ...
```

也可以手动指定原始数据存放目录：

```bash
export EEG_ARENA_RAW_DATA_ROOT=/path/to/raw_datasets
```

运行时若未找到 HDF5 缓存，框架会自动从原始文件构建。

### Step 4：▶️ 开始评测

#### 🧪 单任务验证

可先运行一个单任务，验证环境、预训练权重和数据缓存是否准备完成。以下命令直接使用 `./data/cache`；若缓存不存在，则按 Step 3 中配置的原始数据目录自动构建：

```bash
conda run --no-capture-output -n eeg-arena \
python -u run_single_experiment.py \
  --model reve \
  --variant base \
  --model_size base \
  --dataset bcic_iv_2b \
  --task motor_imagery_left_right_2class \
  --split_mode ws \
  --fold 4 \
  --cache_dir ./data/cache \
  --pipeline datasets/pipelines/reve.py \
  --input_format raw \
  --channel_system 10-5 \
  --epochs 1 \
  --data_ratio 0.01 \
  --batch_size 64 \
  --num_workers 4 \
  --output_dir ./experiments/output/reve_base_smoke_test
```

#### 🚀 批量复现 Protocol I

我们已将论文 Protocol I 的正式训练参数与所有运行配置放置于 [configs/](configs/) 中。完成单任务验证后，以 REVE-base 全参数微调为例，通过以下批量部署命令，可以完整复现我们的评测结果：

```bash
conda run --no-capture-output -n eeg-arena \
python -u run_batch_experiments.py \
  --config configs/protocol_i/eeg_fms/reve__base.json \
  --gpus 0 1 2 3 \
  --processes-per-gpu 1 \
  --poll-interval 5 \
  --output-dir ./experiments/output/protocol_i_reve_base_full_ft
```

### Step 5：📊 查看和可视化评测结果

评测完成后，可使用内置工具汇总实验记录并生成静态可视化 Dashboard：

```bash
python results/build_site.py \
  --suite-dir ./experiments/output/protocol_i_reve_base_full_ft
```

生成的页面位于 `results/output/protocol_i_reve_base_full_ft/dashboard.html`，可直接在浏览器中打开，查看任务状态、评测指标、模型排名、训练曲线及不同数据集和数据划分下的结果对比。

### Step 6：🧩 接入你的模型、数据集

EEG-Arena 提供统一的模型接口、数据集 registry、kernel 与预处理 pipeline，可快速接入并评测新的模型和数据集。模型接入方式见 [模型接口文档](docs/model_interface.md)。

---

## 🏆 加入 Leaderboard

如果您的模型已在 EEG-Arena Benchmark 上完成完整评测，并希望将其加入 [Leaderboard](https://eeg.fm/leaderboards/)，现阶段可以在本仓库提出 Issue，并附上模型说明、评测配置与完整评测结果。

---

## 🧬 模型库

`EEG-Arena` 当前已接入 30 个 EEG FMs 与 25 个监督模型。

<details>
<summary><strong>🧠 展开查看基础模型</strong></summary>

| 模型 | Registry | Variant | 输入格式 | 通道名 `ch_names` | 通道坐标 `ch_positions` | 通道 ID `model_channel_indices` |
|---|---|---|---|---|---|---|
| BENDR | `bendr` | — | `raw` | ✓ | — | — |
| BIOT | `biot` | `shhs-prest-18ch` | `raw` | ✓ | — | — |
| BIOT | `biot` | `prest-16ch` | `raw` | ✓ | — | — |
| BIOT | `biot` | `six-datasets-18ch` | `raw` | ✓ | — | — |
| EEGPT | `eegpt` | — | `patched` | ✓ | — | — |
| LaBraM | `labram` | `base` | `patched` | ✓ | — | ✓ |
| NeuroGPT | `neurogpt` | — | `patched` | ✓ | — | — |
| CBraMod | `cbramod` | — | `patched` | — | — | — |
| CSBrain | `csbrain` | — | `patched` | ✓ | — | — |
| EEGMamba | `eegmamba` | — | `patched` | — | — | — |
| HEAR | `hear` | `base` | `patched` | ✓ | ✓ | — |
| HEAR | `hear` | `large` | `patched` | ✓ | ✓ | — |
| LEAD | `lead` | `base` | `raw` | ✓ | — | — |
| LUNA | `luna` | `base` | `raw` | ✓ | ✓ | — |
| LUNA | `luna` | `large` | `raw` | ✓ | ✓ | — |
| LUNA | `luna` | `huge` | `raw` | ✓ | ✓ | — |
| mdJPT | `mdjpt` | — | `patched` | ✓ | — | — |
| NeuroRVQ | `neurorvq` | — | `patched` | ✓ | — | ✓ |
| REVE | `reve` | `base` | `raw` | — | ✓ | — |
| REVE | `reve` | `large` | `raw` | — | ✓ | — |
| CodeBrain | `codebrain` | — | `patched` | — | — | — |
| ST-EEGFormer | `steegformer` | `small` | `raw` | ✓ | — | ✓ |
| ST-EEGFormer | `steegformer` | `base` | `raw` | ✓ | — | ✓ |
| ST-EEGFormer | `steegformer` | `large` | `raw` | ✓ | — | ✓ |
| SleepFM | `sleepfmclinical` | `sleep-staging` | `raw` | — | — | — |
| SleepFM | `sleepfmclinical` | `diagnosis` | `raw` | — | — | — |
| SleepFM | `sleepfmclinical` | `base-upstream` | `raw` | — | — | — |
| SleepGPT | `sleepgpt` | `large-pretrained` | `raw` | ✓ | — | — |
| TFM | `tfmtokenizer` | `single-chbmit` | `raw` | — | — | — |
| Uni-DMFM | `unidmfm` | `base` | `patched` | — | — | — |

</details>

<details>
<summary><strong>📐 展开查看传统监督基线</strong></summary>

| 模型 | Registry | 输入格式 | 通道名 `ch_names` | 通道坐标 `ch_positions` | 通道 ID `model_channel_indices` |
|---|---|---|---|---|---|
| EEGNet | `eegnet` | `raw` | — | — | — |
| ShallowConvNet | `shallowconvnet` | `raw` | — | — | — |
| DeepConvNet | `deepconvnet` | `raw` | — | — | — |
| EEGConformer | `eegconformer` | `raw` | — | — | — |
| ATCNet | `atcnet` | `raw` | — | — | — |
| CTNet | `ctnet` | `raw` | — | — | — |
| MSVTNet | `msvtnet` | `raw` | — | — | — |
| EEGInceptionERP | `eeginception_erp` | `raw` | — | — | — |
| DBConformer | `dbconformer` | `raw` | — | — | — |
| FBCNet | `fbcnet` | `raw` | — | — | — |
| IFNet | `ifnet` | `raw` | — | — | — |
| EEGWaveNet | `eegwavenet` | `raw` | — | — | — |
| SlimSeiz | `slimseiz` | `raw` | — | — | — |
| MSCFormer | `mscformer` | `raw` | — | — | — |
| TMSA-Net | `tmsanet` | `raw` | — | — | — |
| ADFCNN | `adfcnn` | `raw` | — | — | — |
| EEGNeX | `eegnex` | `raw` | — | — | — |
| BiDANN | `bidann` | `raw` | — | 可选 | — |
| TSception | `tsception` | `raw` | — | — | — |
| HSLT | `hslt` | `raw` | — | 可选 | — |
| DeepSleepNet | `deepsleepnet` | `raw` | — | — | — |
| SleepEEGNet | `sleepeegnet` | `raw` | — | — | — |
| SleepTransformer | `sleeptransformer` | `raw` | — | — | — |
| U-Sleep | `usleep` | `raw` | — | — | — |
| YASA-style baseline | `yasa` | `raw` | — | — | — |

</details>

---

## 🗃️ 数据集与任务库

`EEG-Arena` 当前在 [datasets/__init__.py](datasets/__init__.py) 中内置了 29 个原始数据集，每个数据集下注册了一个或多个具名任务（task），共 74 个 downstream tasks。

同一个数据集可以派生多个下游任务，例如：

- Large-MI：All、Left/Right Hand and Left/Right Leg MI

当前已注册的数据集与任务可按场景概览如下：

| 场景 | 数据集 | 任务 |
| :--- | ---: | ---: |
| 运动想象（MI） | 10 | 25 |
| 睡眠（Sleep） | 4 | 20 |
| 临床（Clinical） | 8 | 20 |
| 认知（Cognitive） | 1 | 1 |
| 听觉注意（Auditory） | 2 | 4 |
| 事件相关电位（ERP） | 2 | 2 |
| 情绪（Emotion） | 2 | 2 |
| **合计** | **29** | **74** |

<details>
<summary><strong>🗂️ 展开查看完整数据集与任务列表</strong></summary>

#### 1.1 运动想象（MI） · 10 个数据集 / 25 个任务

| 数据集 | Registry ID | 任务数 | 注册任务 |
| :--- | :--- | :---: | :--- |
| **BCIC-IV-1** | `bcic_iv_1` | 2 | `motor_imagery_left_right_2class`<br>`motor_imagery_left_foot_2class` |
| **BCIC-IV-2A** | `bcic_iv_2a` | 3 | `motor_imagery_left_right_foot_tongue_4class`<br>`motor_imagery_left_right_2class`<br>`motor_imagery_hand_vs_feet_2class` |
| **BCIC-IV-2B** | `bcic_iv_2b` | 1 | `motor_imagery_left_right_2class` |
| **PhysioNet-MI** | `physionet_eegmmidb` | 3 | `motor_imagery_left_right_both_fists_feet_4class`<br>`motor_imagery_left_right_2class`<br>`motor_imagery_hand_vs_feet_2class` |
| **Large-MI** | `figshare_largemi` | 7 | `motor_imagery_with_rest_11class`<br>`motor_imagery_left_right_2class`<br>`motor_imagery_left_right_passive_3class`<br>`motor_imagery_left_right_hand_leg_4class`<br>`motor_imagery_hand_vs_leg_2class`<br>`motor_imagery_left_right_hand_leg_tongue_5class`<br>`motor_imagery_fingers_5class` |
| **SHU-MI** | `figshare_shudb` | 1 | `motor_imagery_left_right_2class` |
| **Stroke-MI** | `figshare_stroke` | 1 | `motor_imagery_left_right_2class` |
| **Meng2019** | `figshare_meng2019` | 3 | `cursor_control_minus_plus_2class`<br>`cursor_control_minus_plus_rest_3class`<br>`cursor_movement_detection_2class` |
| **HGD** | `other_highgammadataset` | 3 | `motor_imagery_left_right_feet_tongue_4class`<br>`motor_imagery_left_right_2class`<br>`motor_imagery_hand_vs_feet_2class` |
| **OpenBMI** | `other_openbmi` | 1 | `motor_imagery_left_right_2class` |

#### 1.2 睡眠（Sleep） · 4 个数据集 / 20 个任务

| 数据集 | Registry ID | 任务数 | 注册任务 |
| :--- | :--- | :---: | :--- |
| **CAP-Sleep** | `physionet_capslpdb` | 10 | `sleep_staging_30s_6class`<br>`sleep_staging_4s_6class`<br>`sleep_staging_30s_5class`<br>`sleep_staging_4s_5class`<br>`sleep_wake_detection_30s_2class`<br>`sleep_wake_detection_4s_2class`<br>`sleep_wake_nrem_rem_30s_3class`<br>`sleep_wake_nrem_rem_4s_3class`<br>`cap_detection_14s_2class`<br>`cap_subtype_4s_3class` |
| **HMC-Sleep** | `physionet_hmcsleepstaging` | 3 | `sleep_staging_30s_5class`<br>`sleep_wake_detection_30s_2class`<br>`sleep_wake_nrem_rem_30s_3class` |
| **UCDDB-Sleep** | `physionet_ucddb` | 4 | `sleep_staging_30s_6class`<br>`sleep_staging_30s_5class`<br>`sleep_wake_detection_30s_2class`<br>`sleep_wake_nrem_rem_30s_3class` |
| **Sleep-EDF** | `physionet_sleepedfx` | 3 | `sleep_staging_30s_5class`<br>`sleep_wake_detection_30s_2class`<br>`sleep_wake_nrem_rem_30s_3class` |

#### 1.3 临床（Clinical） · 8 个数据集 / 20 个任务

| 数据集 | Registry ID | 任务数 | 注册任务 |
| :--- | :--- | :---: | :--- |
| **TUSL** | `tuh_eeg_slowing` | 2 | `background_slowing_detection_2class`<br>`background_slowing_seizure_3class` |
| **TUEV** | `tuh_eeg_events` | 2 | `epileptiform_event_4class`<br>`epileptiform_event_detection_2class` |
| **MigraineDB** | `other_migrainedb` | 9 | `ssvep_migraine_detection_2class`<br>`ssaep_migraine_detection_2class`<br>`resting_migraine_detection_2class`<br>`ssvep_frequency_2class`<br>`ssaep_frequency_2class`<br>`ssvep_4hz_migraine_detection_2class`<br>`ssvep_6hz_migraine_detection_2class`<br>`ssaep_4hz_migraine_detection_2class`<br>`ssaep_6hz_migraine_detection_2class` |
| **TUAB** | `tuh_eeg_abnormal` | 1 | `abnormal_eeg_detection_2class` |
| **TUEP** | `tuh_eeg_epilepsy` | 1 | `epilepsy_detection_2class` |
| **TUSZ** | `tuh_eeg_seizure` | 2 | `seizure_detection_2class`<br>`seizure_type_9class` |
| **TUAR** | `tuh_eeg_artifact` | 2 | `artifact_detection_2class`<br>`artifact_type_14class` |
| **CHB-MIT** | `physionet_chbmit` | 1 | `seizure_detection_2class` |

#### 1.4 认知（Cognitive） · 1 个数据集 / 1 个任务

| 数据集 | Registry ID | 任务数 | 注册任务 |
| :--- | :--- | :---: | :--- |
| **EEGMat** | `physionet_eegmat` | 1 | `mental_arithmetic_rest_2class` |

#### 1.5 听觉注意（Auditory） · 2 个数据集 / 4 个任务

| 数据集 | Registry ID | 任务数 | 注册任务 |
| :--- | :--- | :---: | :--- |
| **KUL-AAD** | `zenodo_kul` | 1 | `auditory_attention_attended_side_2class` |
| **DTU-AAD** | `zenodo_dtu` | 3 | `auditory_attention_left_right_2class`<br>`auditory_attention_male_female_2class`<br>`auditory_attention_male_female_left_right_4class` |

#### 1.6 事件相关电位（ERP） · 2 个数据集 / 2 个任务

| 数据集 | Registry ID | 任务数 | 注册任务 |
| :--- | :--- | :---: | :--- |
| **ERP-BCI** | `physionet_erpbci` | 1 | `p300_target_detection_2class` |
| **Kaggle-INRIA** | `kaggle_inria` | 1 | `p300_feedback_correctness_2class` |

#### 1.7 情绪（Emotion） · 2 个数据集 / 2 个任务

| 数据集 | Registry ID | 任务数 | 注册任务 |
| :--- | :--- | :---: | :--- |
| **SEED** | `other_seed` | 1 | `emotion_recognition_3class` |
| **FACED** | `other_faced` | 1 | `emotion_recognition_9class` |

</details>

---

## ⚙️ 部署工具

### 1. 单任务运行

[run_single_experiment.py](run_single_experiment.py) 是单任务训练与评估入口脚本。适合用于模型验证、权重排查和单配置复现。

`--dataset` 和 `--task` 为必填参数，`--cache_dir` 指定 HDF5 cache 的存储目录。

例如，运行一个 EEGNet 单任务评测：

```bash
conda run -n eeg-arena python -u run_single_experiment.py \
  --model eegnet \
  --dataset physionet_eegmmidb \
  --task motor_imagery_left_right_both_fists_feet_4class \
  --cache_dir /path/to/cache \
  --pipeline datasets/pipelines/classical_pipeline_with_z_score.py \
  --epoch_param tmin=0.0 tmax=4.0 \
  --epochs 50 \
  --output_dir ./experiments/output/debug_run
```

### 2. 批量任务运行

EEG-Arena 支持在多张 GPU 上批量部署评测任务并完成自动调度，以开展大规模评测；入口脚本为 [run_batch_experiments.py](run_batch_experiments.py)。

批量评测通过预先编写的 JSON suite，再由调度器自动展开并执行。使用以下命令将任务自动分配到指定 GPU：

```bash
conda run --no-capture-output -n eeg-arena \
python -u run_batch_experiments.py \
  --config configs/examples/reve_protocol_i_full_ft.json \
  --gpus 0 1 2 3 \
  --processes-per-gpu 1 \
  --poll-interval 5 \
  --output-dir ./experiments/output/reve_protocol_i_full_ft
```

其中，`--config` 指向 JSON suite。以 Protocol I 中 REVE-base 的 Full-FT 评测配置为例：

<details>
<summary><strong>📄 展开查看 JSON suite 示例</strong></summary>

```jsonc
{
  "models": ["reve"],
  // 指定评测数据集、任务、epoch 等通用运行参数
  "datasets": [
    "bcic_iv_2a:motor_imagery_left_right_foot_tongue_4class",
    "physionet_eegmmidb:motor_imagery_left_right_both_fists_feet_4class"
  ],
  "common_args": {
    "epochs": 50,
    "batch_size": 64,
    "num_workers": 8,
    "channel_system": "10-5",
    "cache_dir": "./data/cache",
    "lr": 0.0001,
    "weight_decay": 0.05,
    "optimizer": "AdamW",
    "label_smoothing": 0.1,
    "encoder_lr_scale": 1.0,
    "classifier_lr_scale": 1.0,
    "early_stopping_patience": 10,
    "save_checkpoints": false,
    // 指定端到端 Full-FT
    "use_pretrained_weights": "True",
    "frozen": "False"
  },
  "sweeps": {
    "seed": [0]
  },
  "split_plan": {
    // 指定 WS 和 CS 两种评测划分
    "ws": {"folds": [4]},
    "cs": {"folds": [4]}
  },
  "model_overrides": {
    "reve": {
      "input_format": "raw",
      "pipeline": "datasets/pipelines/reve.py",
      "pipeline_params": {},
      "variant": "base",
      "model_size": "base"
    }
  }
  // 2 个数据集 × 2 种数据划分方式，共 4 个训练任务
}
```

</details>

可通过 dashboard 监控运行进度：

<details>
<summary><strong>🖥️ 展开查看终端 Dashboard 示例</strong></summary>

```text
EEG-Arena Scheduler Dashboard | Updated 2026-10-11 14:32:08
Progress: 0/4 (0.0%) | completed=0 failed=0 active=4 pending=0
+---------+---------+----------+-----------------+--------+----------+--------------------------------------------------+
| GPU     | State   | PID      | Mem(MiB)        | Util   | Run      | Task                                             |
+---------+---------+----------+-----------------+--------+----------+--------------------------------------------------+
| 0:1     | running | 284101   | 32540/81920     | 94%    | 1m24s    | 0001__reve__bcic_iv_2a__cs_f4__seed0            |
| 1:1     | running | 284102   | 32540/81920     | 94%    | 1m22s    | 0002__reve__bcic_iv_2a__ws_f4__seed0            |
| 2:1     | running | 284103   | 33102/81920     | 91%    | 1m20s    | 0003__reve__physionet_eegmmidb__cs_f4__seed0    |
| 3:1     | running | 284104   | 33102/81920     | 91%    | 1m18s    | 0004__reve__physionet_eegmmidb__ws_f4__seed0    |
+---------+---------+----------+-----------------+--------+----------+--------------------------------------------------+
```

</details>

---

## 📝 引用

如果 EEG-Arena 或 NeuroDoc 对你的研究有所帮助，请引用相应论文。

<details>
<summary><strong>📚 展开查看 BibTeX</strong></summary>

```bibtex
@article{chen2026benchmarking,
  title   = {Benchmarking EEG Foundation Models at Scale: Lessons from 20,000 Evaluations},
  author  = {Chen, Zhige and Peng, Shu and Qin, Chengxuan and Liu, Rui and Yang, Rui and Tan, Kay Chen and Wu, Jibin},
  journal = {arXiv preprint arXiv:2609.32743},
  year    = {2026},
  doi     = {10.48550/arXiv.2609.32743}
}

@article{qin2026neurodoc,
  title   = {EEG Benchmarking Needs a Task Specification Layer: NeuroDoc for Rulebook-Guided, Executable Benchmark Construction},
  author  = {Qin, Chengxuan and Chen, Zhige and Peng, Shu and Yang, Rui and Cui, Jiping and Dong, Yikai and Li, Jun and Peng, Liu and Shang, Zhida and Tang, Mingze and Tan, Kay Chen and Wu, Jibin},
  journal = {arXiv preprint arXiv:2606.22925},
  year    = {2026},
  doi     = {10.48550/arXiv.2606.22925}
}
```

</details>
