<h1 align="center">EEG-Arena: A Large-Scale EEG Model Benchmarking Platform</h1>

<p align="center"><a href="README_ZH.md">中文</a> | <strong>English</strong></p>

<p align="center"><em>Unified data interface and experiment infrastructure for large-scale EEG foundation models</em></p>

<p align="center">
  <a href="docs/model_interface.md"><img alt="MODEL_INTERFACE" src="https://img.shields.io/badge/MODEL__INTERFACE-F3F5F7?style=flat-square&logo=gitbook&logoColor=555555&labelColor=F3F5F7&color=F3F5F7"></a>
  <a href="https://eeg.fm/leaderboards/"><img alt="LEADERBOARD" src="https://img.shields.io/badge/LEADERBOARD-F3F5F7?style=flat-square&logo=starship&logoColor=555555&labelColor=F3F5F7&color=F3F5F7"></a>
  <a href="https://eegarena.eeg.fm/"><img alt="BENCHMARK_RESULTS" src="https://img.shields.io/badge/BENCHMARK__RESULTS-F3F5F7?style=flat-square&logo=plotly&logoColor=555555&labelColor=F3F5F7&color=F3F5F7"></a>
</p>

<p align="center">
  <a href="#-model-zoo"><img alt="30 EEG FMs" src="https://img.shields.io/badge/EEG%20FMs-30-0A7EA4?style=flat-square"></a>
  <a href="#-model-zoo"><img alt="25 supervised baselines" src="https://img.shields.io/badge/Supervised%20Baselines-25-5B8C5A?style=flat-square"></a>
  <a href="datasets/__init__.py"><img alt="29 datasets / 74 tasks" src="https://img.shields.io/badge/Datasets%20%2F%20Tasks-29%20%2F%2074-2D6A4F?style=flat-square"></a>
  <a href="requirements.txt"><img alt="PyTorch 2.5.1" src="https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="#%EF%B8%8F-deployment-tools"><img alt="single-run and batch workflows" src="https://img.shields.io/badge/Single--run%20%2B%20Batch%20Workflows-6C757D?style=flat-square"></a>
</p>

---

## 🚩 News

- **[2026-10-10]** We released the EEG-Arena evaluation toolkit v0.00 together with this user guide.
- **[2026-09-30]** We launched the public [model leaderboard](https://eeg.fm/) and [evaluation results website](https://eegarena.eeg.fm/).
- **[2026-09-30]** The full [EEG-Arena paper](https://arxiv.org/abs/2609.32743) is now available.
- **[2026-09-25]** [NeuroDoc](https://neurodoc.eeg.fm/), the data tool used by EEG-Arena, was accepted to NeurIPS 2026.

## 🧠 Overview

`EEG-Arena` is a unified data and experiment infrastructure for large-scale EEG foundation models. It provides consistent integration and execution interfaces for pretrained models and conventional supervised baselines, along with fair evaluation, scheduling, and reproducibility support at scale.

## ✨ Highlights

- **Data standardization**: unified specifications and data interfaces for large-scale heterogeneous EEG data.
- **Model standardization**: a consistent code convention and integration interface for EEG models.
- **Unified evaluation**: large-scale experiment deployment, scheduling, and result aggregation.
- **Public results**: [Leaderboard](https://eeg.fm/leaderboards/) · [Evaluation Results](https://eegarena.eeg.fm/).
- **Data tooling**: sustainable, large-scale expansion of downstream datasets through [NeuroDoc](https://neurodoc.eeg.fm/).

## 🚀 Quick Start

### Step 1: 🛠️ Install the environment

After cloning the repository, create a Conda environment named `eeg-arena` and install the dependencies from [requirements.txt](requirements.txt):

```bash
conda create -n eeg-arena python=3.12 -y
conda activate eeg-arena
pip install -r requirements.txt
```

EEGMamba additionally requires CUDA extensions. Make sure that Git, GCC 9+, and a CUDA toolkit compatible with your PyTorch installation are available, then run:

```bash
pip install ninja
CAUSAL_CONV1D_FORCE_BUILD=TRUE \
pip install "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0" \
  --no-build-isolation
MAMBA_FORCE_BUILD=TRUE \
pip install "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@v2.2.2" \
  --no-build-isolation
```

### Step 2: 📦 Download and prepare pretrained weights

Place all pretrained weights under `models/Weights/`, preserving the directory names and filenames required by each model configuration.

> [!IMPORTANT]
> The complete set of evaluation weights occupies approximately `10.2 GiB` (`11.0 GB`). Reserve at least `12 GB` of disk space. The expected directory structure is shown below.

<details>
<summary><img alt="📂 Expand the complete weight directory structure" src="https://img.shields.io/badge/%F0%9F%93%82%20Expand-Complete%20Weight%20Layout-0A7EA4?style=flat-square"></summary>

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

Download weights from the official sources below and place them according to the directory structure above:

| Model | Weight source |  | Model | Weight source |
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

### Step 3: 🗂️ Prepare evaluation datasets

EEG-Arena supports two ways to prepare dataset files. For the fastest start, use the first option:

> [!IMPORTANT]
> A full evaluation requires at least **3 TB** of storage for all evaluation caches.

| Data preparation method | Best for | Download | Notes |
|---|---|---|---|
| ⚡ Download standardized data | Quickly reproducing experiments with our standardized downstream data files | [ModelScope](https://www.modelscope.cn/datasets/orangeeg/eeg-arena) | [Dataset documentation](https://eeg.fm/benchmarks/datasets/) |
| 🛠️ Build from raw data | Exporting local raw datasets to the downstream format with EEG-Arena processing scripts | [Download index](https://eeg.fm/datasets/) | HDF5 caches are built on demand and reused |

#### Option 1: ⚡ Use standardized data files

After downloading the standardized files, place them under `./data/cache`, or set another location through `cache_dir` in the run configuration. No raw dataset is required.

The extracted cache should follow this structure:

```text
data/
└── cache/
    └── <dataset>/
        ├── locator.csv
        └── <task>/
            └── <config_key>.hdf5
```

#### Option 2: 🛠️ Build caches from raw data

Place each raw dataset in its own subdirectory. For example, when using `--dataset bcic_iv_2b`, EEG-Arena reads raw files from `data/raw/bcic_iv_2b/`:

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

You can also specify the raw-data root manually:

```bash
export EEG_ARENA_RAW_DATA_ROOT=/path/to/raw_datasets
```

If no matching HDF5 cache exists, the framework automatically builds one from the raw files.

### Step 4: ▶️ Start evaluation

#### 🧪 Single-run smoke test

Start with a single task to verify the environment, pretrained weights, and dataset cache. The command below uses `./data/cache`; when no cache exists, it builds one from the raw-data location configured in Step 3.

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

#### 🚀 Reproduce Protocol I in batch

All official training parameters and run configurations for Protocol I are provided under [configs/](configs/). After the smoke test, the following command reproduces the evaluation using REVE-base full-parameter fine-tuning as an example:

```bash
conda run --no-capture-output -n eeg-arena \
python -u run_batch_experiments.py \
  --config configs/protocol_i/eeg_fms/reve__base.json \
  --gpus 0 1 2 3 \
  --processes-per-gpu 1 \
  --poll-interval 5 \
  --output-dir ./experiments/output/protocol_i_reve_base_full_ft
```

### Step 5: 📊 View and visualize results

After evaluation, aggregate experiment records and generate a static visualization dashboard with the built-in tool:

```bash
python results/build_site.py \
  --suite-dir ./experiments/output/protocol_i_reve_base_full_ft
```

The generated page is written to `results/output/protocol_i_reve_base_full_ft/dashboard.html`. Open it in a browser to inspect task status, metrics, model rankings, training curves, and comparisons across datasets and data splits.

### Step 6: 🧩 Integrate your model or dataset

EEG-Arena provides a unified model interface, dataset registry, kernels, and preprocessing pipelines for quickly integrating and evaluating new models and datasets. See the [model interface documentation](docs/model_interface.md) for model integration.

---

## 🏆 Join the Leaderboard

If your model has completed the full EEG-Arena Benchmark and you would like it to appear on the [Leaderboard](https://eeg.fm/leaderboards/), open an Issue in this repository and include the model description, evaluation configuration, and complete results.

---

## 🧬 Model Zoo

`EEG-Arena` currently integrates 30 EEG FMs and 25 supervised models.

<details>
<summary><strong>🧠 Expand the foundation models</strong></summary>

| Model | Registry | Variant | Input format | Channel names `ch_names` | Channel coordinates `ch_positions` | Channel IDs `model_channel_indices` |
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
<summary><strong>📐 Expand the conventional supervised baselines</strong></summary>

| Model | Registry | Input format | Channel names `ch_names` | Channel coordinates `ch_positions` | Channel IDs `model_channel_indices` |
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
| BiDANN | `bidann` | `raw` | — | Optional | — |
| TSception | `tsception` | `raw` | — | — | — |
| HSLT | `hslt` | `raw` | — | Optional | — |
| DeepSleepNet | `deepsleepnet` | `raw` | — | — | — |
| SleepEEGNet | `sleepeegnet` | `raw` | — | — | — |
| SleepTransformer | `sleeptransformer` | `raw` | — | — | — |
| U-Sleep | `usleep` | `raw` | — | — | — |
| YASA-style baseline | `yasa` | `raw` | — | — | — |

</details>

---

## 🗃️ Dataset and Task Library

`EEG-Arena` currently registers 29 raw datasets in [datasets/__init__.py](datasets/__init__.py). Each dataset contains one or more named tasks, for a total of 74 downstream tasks.

A single dataset may yield multiple downstream tasks. For example:

- Large-MI: All, Left/Right Hand and Left/Right Leg MI

The registered datasets and tasks are summarized by scenario below:

| Scenario | Datasets | Tasks |
| :--- | ---: | ---: |
| Motor imagery (MI) | 10 | 25 |
| Sleep | 4 | 20 |
| Clinical | 8 | 20 |
| Cognitive | 1 | 1 |
| Auditory attention | 2 | 4 |
| Event-related potential (ERP) | 2 | 2 |
| Emotion | 2 | 2 |
| **Total** | **29** | **74** |

<details>
<summary><strong>🗂️ Expand the complete dataset and task list</strong></summary>

#### 1.1 Motor imagery (MI) · 10 datasets / 25 tasks

| Dataset | Registry ID | Tasks | Registered tasks |
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

#### 1.2 Sleep · 4 datasets / 20 tasks

| Dataset | Registry ID | Tasks | Registered tasks |
| :--- | :--- | :---: | :--- |
| **CAP-Sleep** | `physionet_capslpdb` | 10 | `sleep_staging_30s_6class`<br>`sleep_staging_4s_6class`<br>`sleep_staging_30s_5class`<br>`sleep_staging_4s_5class`<br>`sleep_wake_detection_30s_2class`<br>`sleep_wake_detection_4s_2class`<br>`sleep_wake_nrem_rem_30s_3class`<br>`sleep_wake_nrem_rem_4s_3class`<br>`cap_detection_14s_2class`<br>`cap_subtype_4s_3class` |
| **HMC-Sleep** | `physionet_hmcsleepstaging` | 3 | `sleep_staging_30s_5class`<br>`sleep_wake_detection_30s_2class`<br>`sleep_wake_nrem_rem_30s_3class` |
| **UCDDB-Sleep** | `physionet_ucddb` | 4 | `sleep_staging_30s_6class`<br>`sleep_staging_30s_5class`<br>`sleep_wake_detection_30s_2class`<br>`sleep_wake_nrem_rem_30s_3class` |
| **Sleep-EDF** | `physionet_sleepedfx` | 3 | `sleep_staging_30s_5class`<br>`sleep_wake_detection_30s_2class`<br>`sleep_wake_nrem_rem_30s_3class` |

#### 1.3 Clinical · 8 datasets / 20 tasks

| Dataset | Registry ID | Tasks | Registered tasks |
| :--- | :--- | :---: | :--- |
| **TUSL** | `tuh_eeg_slowing` | 2 | `background_slowing_detection_2class`<br>`background_slowing_seizure_3class` |
| **TUEV** | `tuh_eeg_events` | 2 | `epileptiform_event_4class`<br>`epileptiform_event_detection_2class` |
| **MigraineDB** | `other_migrainedb` | 9 | `ssvep_migraine_detection_2class`<br>`ssaep_migraine_detection_2class`<br>`resting_migraine_detection_2class`<br>`ssvep_frequency_2class`<br>`ssaep_frequency_2class`<br>`ssvep_4hz_migraine_detection_2class`<br>`ssvep_6hz_migraine_detection_2class`<br>`ssaep_4hz_migraine_detection_2class`<br>`ssaep_6hz_migraine_detection_2class` |
| **TUAB** | `tuh_eeg_abnormal` | 1 | `abnormal_eeg_detection_2class` |
| **TUEP** | `tuh_eeg_epilepsy` | 1 | `epilepsy_detection_2class` |
| **TUSZ** | `tuh_eeg_seizure` | 2 | `seizure_detection_2class`<br>`seizure_type_9class` |
| **TUAR** | `tuh_eeg_artifact` | 2 | `artifact_detection_2class`<br>`artifact_type_14class` |
| **CHB-MIT** | `physionet_chbmit` | 1 | `seizure_detection_2class` |

#### 1.4 Cognitive · 1 dataset / 1 task

| Dataset | Registry ID | Tasks | Registered tasks |
| :--- | :--- | :---: | :--- |
| **EEGMat** | `physionet_eegmat` | 1 | `mental_arithmetic_rest_2class` |

#### 1.5 Auditory attention · 2 datasets / 4 tasks

| Dataset | Registry ID | Tasks | Registered tasks |
| :--- | :--- | :---: | :--- |
| **KUL-AAD** | `zenodo_kul` | 1 | `auditory_attention_attended_side_2class` |
| **DTU-AAD** | `zenodo_dtu` | 3 | `auditory_attention_left_right_2class`<br>`auditory_attention_male_female_2class`<br>`auditory_attention_male_female_left_right_4class` |

#### 1.6 Event-related potential (ERP) · 2 datasets / 2 tasks

| Dataset | Registry ID | Tasks | Registered tasks |
| :--- | :--- | :---: | :--- |
| **ERP-BCI** | `physionet_erpbci` | 1 | `p300_target_detection_2class` |
| **Kaggle-INRIA** | `kaggle_inria` | 1 | `p300_feedback_correctness_2class` |

#### 1.7 Emotion · 2 datasets / 2 tasks

| Dataset | Registry ID | Tasks | Registered tasks |
| :--- | :--- | :---: | :--- |
| **SEED** | `other_seed` | 1 | `emotion_recognition_3class` |
| **FACED** | `other_faced` | 1 | `emotion_recognition_9class` |

</details>

---

## ⚙️ Deployment Tools

### 1. Single-run evaluation

[run_single_experiment.py](run_single_experiment.py) is the entry point for training and evaluating a single task. It is suitable for model validation, weight troubleshooting, and reproducing an individual configuration.

`--dataset` and `--task` are required. `--cache_dir` specifies where the HDF5 cache is stored.

For example, run a single-task EEGNet evaluation:

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

### 2. Batch evaluation

EEG-Arena supports large-scale evaluation by deploying tasks across multiple GPUs and scheduling them automatically. The entry point is [run_batch_experiments.py](run_batch_experiments.py).

Batch evaluations are defined in a JSON suite and expanded automatically by the scheduler. Use the following command to distribute tasks across the selected GPUs:

```bash
conda run --no-capture-output -n eeg-arena \
python -u run_batch_experiments.py \
  --config configs/examples/reve_protocol_i_full_ft.json \
  --gpus 0 1 2 3 \
  --processes-per-gpu 1 \
  --poll-interval 5 \
  --output-dir ./experiments/output/reve_protocol_i_full_ft
```

Here, `--config` points to the JSON suite. The following example represents a REVE-base Full-FT evaluation under Protocol I:

<details>
<summary><strong>📄 View the JSON suite example</strong></summary>

```jsonc
{
  "models": ["reve"],
  // Specify evaluation datasets, tasks, epochs, and other shared run parameters
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
    // Enable end-to-end full-parameter fine-tuning
    "use_pretrained_weights": "True",
    "frozen": "False"
  },
  "sweeps": {
    "seed": [0]
  },
  "split_plan": {
    // Evaluate both WS and CS splits
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
  // 2 datasets × 2 split modes = 4 training tasks
}
```

</details>

Monitor progress through the dashboard:

<details>
<summary><strong>🖥️ View a terminal dashboard example</strong></summary>

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

## 📝 Citation

If EEG-Arena or NeuroDoc helps your research, please cite the corresponding paper.

<details>
<summary><strong>📚 View BibTeX</strong></summary>

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
