"""
EEG Channel System Configurations
Defines multiple standard EEG channel systems for selection
"""
from typing import List, Dict, Optional


PUBLIC_CHANNEL_SYSTEM_CHOICES = ('19', '10-20', '10-10', '10-5', 'dynamic', 'custom')

# Strict 19-channel classic 10-20 space used by most EEG FM models (CBraMod,
# BENDR, LEAD, LaBraM, etc.).  Both the legacy IFSECN names (T3/T4/T5/T6) and
# the current IFCN aliases (T7/T8/P7/P8) are included so that datasets using
# either convention are matched correctly.
CHANNEL_SYSTEM_19 = [
    'EEG:Fp1', 'EEG:Fp2',
    'EEG:F7', 'EEG:F3', 'EEG:Fz', 'EEG:F4', 'EEG:F8',
    'EEG:T3', 'EEG:T7', 'EEG:C3', 'EEG:Cz', 'EEG:C4', 'EEG:T4', 'EEG:T8',
    'EEG:T5', 'EEG:P7', 'EEG:P3', 'EEG:Pz', 'EEG:P4', 'EEG:T6', 'EEG:P8',
    'EEG:O1', 'EEG:O2',
]

# Standard 10-20 system (international standard, no extensions).
# Covers the 19 classic electrodes plus Fpz/Oz midline positions and mastoid/
# auricular references (A1/A2/M1/M2) and T1/T2/Iz that appear in some 10-20
# recordings.  Temporal aliases (T3↔T7, T4↔T8, T5↔P7, T6↔P8) are kept for
# cross-dataset compatibility.  AF, FC, CP, PO intermediate positions are NOT
# included here — those belong to the 10-10 system.
CHANNEL_SYSTEM_10_20 = [
    # Frontopolar
    'EEG:Fp1', 'EEG:Fpz', 'EEG:Fp2',
    # Frontal
    'EEG:F7', 'EEG:F3', 'EEG:Fz', 'EEG:F4', 'EEG:F8',
    # Central / temporal (legacy + current naming)
    'EEG:T3', 'EEG:T7', 'EEG:C3', 'EEG:Cz', 'EEG:C4', 'EEG:T4', 'EEG:T8',
    # Parietal / temporal (legacy + current naming)
    'EEG:T5', 'EEG:P7', 'EEG:P3', 'EEG:Pz', 'EEG:P4', 'EEG:T6', 'EEG:P8',
    # Occipital
    'EEG:O1', 'EEG:Oz', 'EEG:O2',
    # Mastoid / auricular references
    'EEG:A1', 'EEG:A2', 'EEG:M1', 'EEG:M2',
    # Inferior / nasion sites occasionally present in 10-20 recordings
    'EEG:T1', 'EEG:T2', 'EEG:Iz',
]

# 10-10 system channels (extended from 10-20)
# Includes additional intermediate positions
CHANNEL_SYSTEM_10_10 = [
    # Frontal polar
    'EEG:Fp1', 'EEG:Fpz', 'EEG:Fp2',
    # Anterior frontal
    'EEG:AF9', 'EEG:AF7', 'EEG:AF5', 'EEG:AF3', 'EEG:AF1', 'EEG:AFz', 'EEG:AF2', 'EEG:AF4', 'EEG:AF6', 'EEG:AF8', 'EEG:AF10',
    # Frontal
    'EEG:F9', 'EEG:F7', 'EEG:F5', 'EEG:F3', 'EEG:F1', 'EEG:Fz', 'EEG:F2', 'EEG:F4', 'EEG:F6', 'EEG:F8', 'EEG:F10',
    # Frontotemporal
    'EEG:FT9', 'EEG:FT7', 'EEG:FC5', 'EEG:FC3', 'EEG:FC1', 'EEG:FCz', 'EEG:FC2', 'EEG:FC4', 'EEG:FC6', 'EEG:FT8', 'EEG:FT10',
    # Temporal
    'EEG:T9', 'EEG:T7', 'EEG:C5', 'EEG:C3', 'EEG:C1', 'EEG:Cz', 'EEG:C2', 'EEG:C4', 'EEG:C6', 'EEG:T8', 'EEG:T10',
    # Temporoparietal
    'EEG:TP9', 'EEG:TP7', 'EEG:CP5', 'EEG:CP3', 'EEG:CP1', 'EEG:CPz', 'EEG:CP2', 'EEG:CP4', 'EEG:CP6', 'EEG:TP8', 'EEG:TP10',
    # Parietal
    'EEG:P9', 'EEG:P7', 'EEG:P5', 'EEG:P3', 'EEG:P1', 'EEG:Pz', 'EEG:P2', 'EEG:P4', 'EEG:P6', 'EEG:P8', 'EEG:P10',
    # Parieto-occipital
    'EEG:PO9', 'EEG:PO7', 'EEG:PO5', 'EEG:PO3', 'EEG:PO1', 'EEG:POz', 'EEG:PO2', 'EEG:PO4', 'EEG:PO6', 'EEG:PO8', 'EEG:PO10',
    # Occipital
    'EEG:O1', 'EEG:Oz', 'EEG:O2', 'EEG:O9',
    # Other standard channels
    'EEG:Iz', 'EEG:O10',
    # Classic temporal channels
    'EEG:T3', 'EEG:T5', 'EEG:T4', 'EEG:T6',
    # Mastoids/Auriculars
    'EEG:M1', 'EEG:M2', 'EEG:A1', 'EEG:A2',
    # Central posterior
    'EEG:CCP1', 'EEG:CCP2', 'EEG:CCP3', 'EEG:CCP4', 'EEG:CCP5', 'EEG:CCP6',
    # Additional standard channels
    'EEG:T1', 'EEG:T2',
    # Temporal intermediate
    'EEG:FTT9h', 'EEG:TTP7h', 'EEG:TPP9h', 'EEG:FTT10h', 'EEG:TPP8h', 'EEG:TPP10h',
    # Additional 10-10 specific (intermediate positions)
    'EEG:F10', 'EEG:F6', 'EEG:F4', 'EEG:F2', 'EEG:Fz',
    'EEG:FC4', 'EEG:FC2', 'EEG:FCz',
    'EEG:C4', 'EEG:C2', 'EEG:Cz',
    'EEG:CP4', 'EEG:CP2', 'EEG:CPz',
    'EEG:P4', 'EEG:P2', 'EEG:Pz',
    'EEG:PO4', 'EEG:PO2', 'EEG:POz',
]

# 10-5 system channels (maximum coverage)
# Comprehensive 10-5 electrode placement system
CHANNEL_SYSTEM_10_5 = [
    # Frontopolar
    'EEG:Fp1', 'EEG:Fpz', 'EEG:Fp2',
    # Anterior frontal
    'EEG:AF9', 'EEG:AF7', 'EEG:AF5', 'EEG:AF3', 'EEG:AF1', 'EEG:AFz', 'EEG:AF2', 'EEG:AF4', 'EEG:AF6', 'EEG:AF8', 'EEG:AF10',
    # Frontolateral
    'EEG:F9', 'EEG:F7', 'EEG:F5', 'EEG:F3', 'EEG:F1', 'EEG:Fz', 'EEG:F2', 'EEG:F4', 'EEG:F6', 'EEG:F8', 'EEG:F10',
    # Frontotemporal
    'EEG:FT9', 'EEG:FT7', 'EEG:FC5', 'EEG:FC3', 'EEG:FC1', 'EEG:FCz', 'EEG:FC2', 'EEG:FC4', 'EEG:FC6', 'EEG:FT8', 'EEG:FT10',
    # Temporal
    'EEG:T9', 'EEG:T7', 'EEG:C5', 'EEG:C3', 'EEG:C1', 'EEG:Cz', 'EEG:C2', 'EEG:C4', 'EEG:C6', 'EEG:T8', 'EEG:T10',
    # Temporoparietal
    'EEG:TP9', 'EEG:TP7', 'EEG:CP5', 'EEG:CP3', 'EEG:CP1', 'EEG:CPz', 'EEG:CP2', 'EEG:CP4', 'EEG:CP6', 'EEG:TP8', 'EEG:TP10',
    # Parietal
    'EEG:P9', 'EEG:P7', 'EEG:P5', 'EEG:P3', 'EEG:P1', 'EEG:Pz', 'EEG:P2', 'EEG:P4', 'EEG:P6', 'EEG:P8', 'EEG:P10',
    # Parieto-occipital
    'EEG:PO9', 'EEG:PO7', 'EEG:PO5', 'EEG:PO3', 'EEG:PO1', 'EEG:POz', 'EEG:PO2', 'EEG:PO4', 'EEG:PO6', 'EEG:PO8', 'EEG:PO10',
    # Occipital
    'EEG:O1', 'EEG:Oz', 'EEG:O2', 'EEG:O9',
    # Other
    'EEG:Iz', 'EEG:O10',
    # Classic
    'EEG:T3', 'EEG:T5', 'EEG:T4', 'EEG:T6',
    'EEG:M1', 'EEG:M2', 'EEG:A1', 'EEG:A2',
    # Central posterior
    'EEG:CCP1', 'EEG:CCP2', 'EEG:CCP3', 'EEG:CCP4', 'EEG:CCP5', 'EEG:CCP6',
    # Additional
    'EEG:T1', 'EEG:T2',
    'EEG:FTT9h', 'EEG:TTP7h', 'EEG:TPP9h', 'EEG:FTT10h', 'EEG:TPP8h', 'EEG:TPP10h',
    # 10-10 extended
    'EEG:F10', 'EEG:F8', 'EEG:F6', 'EEG:F4', 'EEG:F2',
    'EEG:FC6', 'EEG:FC4', 'EEG:FC2', 'EEG:FCz',
    'EEG:C6', 'EEG:C4', 'EEG:C2', 'EEG:Cz',
    'EEG:CP6', 'EEG:CP4', 'EEG:CP2', 'EEG:CPz',
    'EEG:P10', 'EEG:P8', 'EEG:P6', 'EEG:P4', 'EEG:P2', 'EEG:Pz',
    'EEG:PO10', 'EEG:PO8', 'EEG:PO6', 'EEG:PO4', 'EEG:PO2', 'EEG:POz',
    # 10-5 specific additional intermediate positions
    'EEG:AFp9', 'EEG:AFp7', 'EEG:AFp5', 'EEG:AFp3', 'EEG:AFp1', 'EEG:AFpz', 'EEG:AFp2', 'EEG:AFp4', 'EEG:AFp6', 'EEG:AFp8', 'EEG:AFp10',
    'EEG:FpO9', 'EEG:FpO7', 'EEG:FpO5', 'EEG:FpO3', 'EEG:FpO1', 'EEG:FpOz', 'EEG:FpO2', 'EEG:FpO4', 'EEG:FpO6', 'EEG:FpO8', 'EEG:FpO10',
    'EEG:POO9', 'EEG:POO7', 'EEG:POO5', 'EEG:POO3', 'EEG:POO1', 'EEG:POOz', 'EEG:POO2', 'EEG:POO4', 'EEG:POO6', 'EEG:POO8', 'EEG:POO10',
    'EEG:I1', 'EEG:I2', 'EEG:OI1', 'EEG:OI2',
]

# All available channel systems
CHANNEL_SYSTEMS: Dict[str, List[str]] = {
    '19': CHANNEL_SYSTEM_19,
    '10-20': CHANNEL_SYSTEM_10_20,
    '10-10': CHANNEL_SYSTEM_10_10,
    '10-5': CHANNEL_SYSTEM_10_5,
    'capslpdb-5ch': [
        'EEG:Fp2-F4',
        'EEG:F4-C4',
        'EEG:C4-P4',
        'EEG:P4-O2',
        'EEG:C4-A1',
    ],
    'chbmit-bipolar-17ch': [
        'EEG:FP1-F7',
        'EEG:F7-T7',
        'EEG:T7-P7',
        'EEG:P7-O1',
        'EEG:FP1-F3',
        'EEG:F3-C3',
        'EEG:C3-P3',
        'EEG:P3-O1',
        'EEG:FP2-F4',
        'EEG:F4-C4',
        'EEG:C4-P4',
        'EEG:P4-O2',
        'EEG:FP2-F8',
        'EEG:F8-T8',
        'EEG:FZ-CZ',
        'EEG:CZ-PZ',
        'EEG:P8-O2',
    ],
    'hmcsleep-4ch': [
        'EEG:F4-M1',
        'EEG:C4-M1',
        'EEG:O2-M1',
        'EEG:C3-M2',
    ],
    'sleepedfx-2ch': [
        'EEG:Fpz-Cz',
        'EEG:Pz-Oz',
    ],
    'ucddb-2ch': [
        'EEG:C3-A2',
        'EEG:C4-A1',
    ],
}

# ── Model-specific input channel spaces ──────────────────────────────────────
# Each model may have its own pretrained channel space.
# Models not listed here should use the user-selected channel system directly.

# BENDR: 19 scalp EEG channels from the compact UI 10/20 space. The pretrained
# encoder has a 20th relative-amplitude input feature, which is not a scalp
# electrode and is intentionally not included here.
MODEL_CHANNEL_SPACE_BENDR = [
    'EEG:Fp1', 'EEG:Fp2',
    'EEG:F7', 'EEG:F3', 'EEG:Fz', 'EEG:F4', 'EEG:F8',
    'EEG:T3', 'EEG:C3', 'EEG:Cz', 'EEG:C4', 'EEG:T4',
    'EEG:T5', 'EEG:P3', 'EEG:Pz', 'EEG:P4', 'EEG:T6',
    'EEG:O1', 'EEG:O2',
]

# BIOT: variant-specific bipolar channel spaces. The 18-channel variants add
# two sleep-reference derivations on top of the PREST 16-channel montage.
MODEL_CHANNEL_SPACE_BIOT_PREST_16CH = [
    'EEG:Fp1-F7', 'EEG:F7-T7', 'EEG:T7-P7', 'EEG:P7-O1',
    'EEG:Fp2-F8', 'EEG:F8-T8', 'EEG:T8-P8', 'EEG:P8-O2',
    'EEG:Fp1-F3', 'EEG:F3-C3', 'EEG:C3-P3', 'EEG:P3-O1',
    'EEG:Fp2-F4', 'EEG:F4-C4', 'EEG:C4-P4', 'EEG:P4-O2',
]
MODEL_CHANNEL_SPACE_BIOT_18CH = MODEL_CHANNEL_SPACE_BIOT_PREST_16CH + [
    'EEG:C3-A2', 'EEG:C4-A1',
]

# LEAD: paper preprocessing aligns recordings to the same 19-channel 10-20
# montage used by BENDR/CBraMod-style datasets.
MODEL_CHANNEL_SPACE_LEAD = MODEL_CHANNEL_SPACE_BENDR.copy()

# mdJPT: standard 60-channel 10-20 configuration used by the unified Matlab
# preprocessing pipeline.
MODEL_CHANNEL_SPACE_MDJPT = [
    'EEG:Fp1', 'EEG:Fpz', 'EEG:Fp2',
    'EEG:AF3', 'EEG:AF4',
    'EEG:F7', 'EEG:F5', 'EEG:F3', 'EEG:F1', 'EEG:Fz', 'EEG:F2', 'EEG:F4', 'EEG:F6', 'EEG:F8',
    'EEG:FT7', 'EEG:FC5', 'EEG:FC3', 'EEG:FC1', 'EEG:FCz', 'EEG:FC2', 'EEG:FC4', 'EEG:FC6', 'EEG:FT8',
    'EEG:T7', 'EEG:C5', 'EEG:C3', 'EEG:C1', 'EEG:Cz', 'EEG:C2', 'EEG:C4', 'EEG:C6', 'EEG:T8',
    'EEG:TP7', 'EEG:CP5', 'EEG:CP3', 'EEG:CP1', 'EEG:CPz', 'EEG:CP2', 'EEG:CP4', 'EEG:CP6', 'EEG:TP8',
    'EEG:P7', 'EEG:P5', 'EEG:P3', 'EEG:P1', 'EEG:Pz', 'EEG:P2', 'EEG:P4', 'EEG:P6', 'EEG:P8',
    'EEG:PO7', 'EEG:PO5', 'EEG:PO3', 'EEG:POz', 'EEG:PO4', 'EEG:PO6', 'EEG:PO8',
    'EEG:O1', 'EEG:Oz', 'EEG:O2',
]

# NeuroGPT: 22-channel extended 10-20 pretraining layout.
MODEL_CHANNEL_SPACE_NEUROGPT = [
    'EEG:Fp1', 'EEG:Fp2',
    'EEG:F7', 'EEG:F3', 'EEG:Fz', 'EEG:F4', 'EEG:F8',
    'EEG:T1', 'EEG:T3', 'EEG:C3', 'EEG:Cz', 'EEG:C4', 'EEG:T4', 'EEG:T2',
    'EEG:T5', 'EEG:P3', 'EEG:Pz', 'EEG:P4', 'EEG:T6',
    'EEG:O1', 'EEG:Oz', 'EEG:O2',
]

# EEGPT: 58 channels (from EEGPT pretrained CHANNEL_DICT)
MODEL_CHANNEL_SPACE_EEGPT = [
    'EEG:Fp1', 'EEG:Fpz', 'EEG:Fp2',
    'EEG:AF7', 'EEG:AF3', 'EEG:AF4', 'EEG:AF8',
    'EEG:F7', 'EEG:F5', 'EEG:F3', 'EEG:F1', 'EEG:Fz', 'EEG:F2', 'EEG:F4', 'EEG:F6', 'EEG:F8',
    'EEG:FT7', 'EEG:FC5', 'EEG:FC3', 'EEG:FC1', 'EEG:FCz', 'EEG:FC2', 'EEG:FC4', 'EEG:FC6', 'EEG:FT8',
    'EEG:T7', 'EEG:C5', 'EEG:C3', 'EEG:C1', 'EEG:Cz', 'EEG:C2', 'EEG:C4', 'EEG:C6', 'EEG:T8',
    'EEG:TP7', 'EEG:CP5', 'EEG:CP3', 'EEG:CP1', 'EEG:CPz', 'EEG:CP2', 'EEG:CP4', 'EEG:CP6', 'EEG:TP8',
    'EEG:P7', 'EEG:P5', 'EEG:P3', 'EEG:P1', 'EEG:Pz', 'EEG:P2', 'EEG:P4', 'EEG:P6', 'EEG:P8',
    'EEG:PO7', 'EEG:PO5', 'EEG:PO3', 'EEG:POz', 'EEG:PO4', 'EEG:PO6', 'EEG:PO8',
    'EEG:O1', 'EEG:Oz', 'EEG:O2',
]

# ST-EEGFormer: official 142-channel pretrained vocabulary from
# `pretrain/senloc_file/sen_chan_idx.pkl`. The upstream encoder uses a
# 145-slot embedding table, but published channel indices occupy 0-141.
MODEL_CHANNEL_SPACE_STEEGFORMER = [
    'EEG:C1', 'EEG:Pz', 'EEG:C4', 'EEG:F6', 'EEG:FTT8h', 'EEG:Oz', 'EEG:Fp1',
    'EEG:FCC5h', 'EEG:TPP8h', 'EEG:CPP6h', 'EEG:C2', 'EEG:F4', 'EEG:OI2h',
    'EEG:AF4', 'EEG:FCz', 'EEG:CCP6h', 'EEG:TP8', 'EEG:POO10h', 'EEG:FC1',
    'EEG:FC6', 'EEG:C5', 'EEG:P8', 'EEG:FT8', 'EEG:P6', 'EEG:P9', 'EEG:Fz',
    'EEG:AFF1', 'EEG:TPP10h', 'EEG:AFF2', 'EEG:P10', 'EEG:CPP2h', 'EEG:M1',
    'EEG:FCC6h', 'EEG:FTT7h', 'EEG:FC2', 'EEG:PPO2', 'EEG:AFp3h', 'EEG:AF7',
    'EEG:PO10', 'EEG:AF8', 'EEG:CPP1h', 'EEG:P7', 'EEG:F1', 'EEG:AFp4h',
    'EEG:PO9', 'EEG:FT9', 'EEG:CP2', 'EEG:Iz', 'EEG:FCC1h', 'EEG:FC5',
    'EEG:T5', 'EEG:CP5', 'EEG:CP6', 'EEG:FFC5h', 'EEG:F2', 'EEG:M2',
    'EEG:POO9h', 'EEG:AFF5h', 'EEG:PO4', 'EEG:POO3h', 'EEG:Fp2', 'EEG:T3',
    'EEG:CP4', 'EEG:POz', 'EEG:TTP7h', 'EEG:T7', 'EEG:A2', 'EEG:CCP4h',
    'EEG:T8', 'EEG:PPO10h', 'EEG:FC3', 'EEG:F3', 'EEG:F5', 'EEG:A1',
    'EEG:P3', 'EEG:FC4', 'EEG:FCC2h', 'EEG:FFC6h', 'EEG:FFT8h', 'EEG:CCP2h',
    'EEG:CPP4h', 'EEG:T6', 'EEG:FTT9h', 'EEG:PPO6h', 'EEG:CP3', 'EEG:CP1',
    'EEG:AF3', 'EEG:FT10', 'EEG:OI1h', 'EEG:TPP9h', 'EEG:P5', 'EEG:I2',
    'EEG:CCP1h', 'EEG:T4', 'EEG:CCP3h', 'EEG:O1', 'EEG:PO5', 'EEG:PPO9h',
    'EEG:PPO5h', 'EEG:P1', 'EEG:AFz', 'EEG:PO6', 'EEG:PO3', 'EEG:O2',
    'EEG:CPP5h', 'EEG:FFC1h', 'EEG:FCC4h', 'EEG:FFT7h', 'EEG:FFC2h',
    'EEG:FFC4h', 'EEG:Cz', 'EEG:TP7', 'EEG:Fpz', 'EEG:FTT10h', 'EEG:PO7',
    'EEG:CPP3h', 'EEG:P4', 'EEG:P2', 'EEG:F8', 'EEG:CPz', 'EEG:FCC3h',
    'EEG:FFC3h', 'EEG:FT7', 'EEG:I1', 'EEG:TTP8h', 'EEG:AFF6h', 'EEG:CCP5h',
    'EEG:C6', 'EEG:PPO1', 'EEG:PO8', 'EEG:C3', 'EEG:POO4h', 'EEG:TPP7h',
    'EEG:F7', 'EEG:T9', 'EEG:TP9', 'EEG:T10', 'EEG:TP10', 'EEG:POO1',
    'EEG:POO2', 'EEG:PPO1h', 'EEG:PPO2h',
]

# LaBraM: 136 channels (120 single-electrode + 16 bipolar) from pretrained model
# pos_embed has 128+1 slots; single electrodes (indices 0-119) fit within 128 slots.
# Bipolar channels (indices 120-135) won't match standard dataset electrodes.
MODEL_CHANNEL_SPACE_LABRAM = [
    # Single electrodes (120 channels, indices 0-119)
    'EEG:Fp1', 'EEG:Fpz', 'EEG:Fp2',
    'EEG:AF9', 'EEG:AF7', 'EEG:AF5', 'EEG:AF3', 'EEG:AF1', 'EEG:AFz', 'EEG:AF2', 'EEG:AF4', 'EEG:AF6', 'EEG:AF8', 'EEG:AF10',
    'EEG:F9', 'EEG:F7', 'EEG:F5', 'EEG:F3', 'EEG:F1', 'EEG:Fz', 'EEG:F2', 'EEG:F4', 'EEG:F6', 'EEG:F8', 'EEG:F10',
    'EEG:FT9', 'EEG:FT7', 'EEG:FC5', 'EEG:FC3', 'EEG:FC1', 'EEG:FCz', 'EEG:FC2', 'EEG:FC4', 'EEG:FC6', 'EEG:FT8', 'EEG:FT10',
    'EEG:T9', 'EEG:T7', 'EEG:C5', 'EEG:C3', 'EEG:C1', 'EEG:Cz', 'EEG:C2', 'EEG:C4', 'EEG:C6', 'EEG:T8', 'EEG:T10',
    'EEG:TP9', 'EEG:TP7', 'EEG:CP5', 'EEG:CP3', 'EEG:CP1', 'EEG:CPz', 'EEG:CP2', 'EEG:CP4', 'EEG:CP6', 'EEG:TP8', 'EEG:TP10',
    'EEG:P9', 'EEG:P7', 'EEG:P5', 'EEG:P3', 'EEG:P1', 'EEG:Pz', 'EEG:P2', 'EEG:P4', 'EEG:P6', 'EEG:P8', 'EEG:P10',
    'EEG:PO9', 'EEG:PO7', 'EEG:PO5', 'EEG:PO3', 'EEG:PO1', 'EEG:POz', 'EEG:PO2', 'EEG:PO4', 'EEG:PO6', 'EEG:PO8', 'EEG:PO10',
    'EEG:O1', 'EEG:Oz', 'EEG:O2', 'EEG:O9', 'EEG:CB1', 'EEG:CB2',
    'EEG:Iz', 'EEG:O10', 'EEG:T3', 'EEG:T5', 'EEG:T4', 'EEG:T6', 'EEG:M1', 'EEG:M2', 'EEG:A1', 'EEG:A2',
    'EEG:CFC1', 'EEG:CFC2', 'EEG:CFC3', 'EEG:CFC4', 'EEG:CFC5', 'EEG:CFC6', 'EEG:CFC7', 'EEG:CFC8',
    'EEG:CCP1', 'EEG:CCP2', 'EEG:CCP3', 'EEG:CCP4', 'EEG:CCP5', 'EEG:CCP6', 'EEG:CCP7', 'EEG:CCP8',
    'EEG:T1', 'EEG:T2', 'EEG:FTT9h', 'EEG:TTP7h', 'EEG:TPP9h', 'EEG:FTT10h', 'EEG:TPP8h', 'EEG:TPP10h',
    # Bipolar channels (16 channels, indices 120-135)
    'EEG:Fp1-F7', 'EEG:F7-T7', 'EEG:T7-P7', 'EEG:P7-O1',
    'EEG:Fp2-F8', 'EEG:F8-T8', 'EEG:T8-P8', 'EEG:P8-O2',
    'EEG:Fp1-F3', 'EEG:F3-C3', 'EEG:C3-P3', 'EEG:P3-O1',
    'EEG:Fp2-F4', 'EEG:F4-C4', 'EEG:C4-P4', 'EEG:P4-O2',
]

# NeuroLM: official tokenizer/GPT channel vocabulary from upstream `standard_1020`
# plus the extra `pad`, `I1`, and `I2` tokens used in the original code.
MODEL_CHANNEL_SPACE_NEUROLM = [
    'EEG:Fp1', 'EEG:Fpz', 'EEG:Fp2',
    'EEG:AF9', 'EEG:AF7', 'EEG:AF5', 'EEG:AF3', 'EEG:AF1', 'EEG:AFz', 'EEG:AF2', 'EEG:AF4', 'EEG:AF6', 'EEG:AF8', 'EEG:AF10',
    'EEG:F9', 'EEG:F7', 'EEG:F5', 'EEG:F3', 'EEG:F1', 'EEG:Fz', 'EEG:F2', 'EEG:F4', 'EEG:F6', 'EEG:F8', 'EEG:F10',
    'EEG:FT9', 'EEG:FT7', 'EEG:FC5', 'EEG:FC3', 'EEG:FC1', 'EEG:FCz', 'EEG:FC2', 'EEG:FC4', 'EEG:FC6', 'EEG:FT8', 'EEG:FT10',
    'EEG:T9', 'EEG:T7', 'EEG:C5', 'EEG:C3', 'EEG:C1', 'EEG:Cz', 'EEG:C2', 'EEG:C4', 'EEG:C6', 'EEG:T8', 'EEG:T10',
    'EEG:TP9', 'EEG:TP7', 'EEG:CP5', 'EEG:CP3', 'EEG:CP1', 'EEG:CPz', 'EEG:CP2', 'EEG:CP4', 'EEG:CP6', 'EEG:TP8', 'EEG:TP10',
    'EEG:P9', 'EEG:P7', 'EEG:P5', 'EEG:P3', 'EEG:P1', 'EEG:Pz', 'EEG:P2', 'EEG:P4', 'EEG:P6', 'EEG:P8', 'EEG:P10',
    'EEG:PO9', 'EEG:PO7', 'EEG:PO5', 'EEG:PO3', 'EEG:PO1', 'EEG:POz', 'EEG:PO2', 'EEG:PO4', 'EEG:PO6', 'EEG:PO8', 'EEG:PO10',
    'EEG:O1', 'EEG:Oz', 'EEG:O2', 'EEG:O9', 'EEG:CB1', 'EEG:CB2',
    'EEG:Iz', 'EEG:O10', 'EEG:T3', 'EEG:T5', 'EEG:T4', 'EEG:T6', 'EEG:M1', 'EEG:M2', 'EEG:A1', 'EEG:A2',
    'EEG:CFC1', 'EEG:CFC2', 'EEG:CFC3', 'EEG:CFC4', 'EEG:CFC5', 'EEG:CFC6', 'EEG:CFC7', 'EEG:CFC8',
    'EEG:CCP1', 'EEG:CCP2', 'EEG:CCP3', 'EEG:CCP4', 'EEG:CCP5', 'EEG:CCP6', 'EEG:CCP7', 'EEG:CCP8',
    'EEG:T1', 'EEG:T2', 'EEG:FTT9h', 'EEG:TTP7h', 'EEG:TPP9h', 'EEG:FTT10h', 'EEG:TPP8h', 'EEG:TPP10h',
    'EEG:Fp1-F7', 'EEG:F7-T7', 'EEG:T7-P7', 'EEG:P7-O1',
    'EEG:Fp2-F8', 'EEG:F8-T8', 'EEG:T8-P8', 'EEG:P8-O2',
    'EEG:Fp1-F3', 'EEG:F3-C3', 'EEG:C3-P3', 'EEG:P3-O1',
    'EEG:Fp2-F4', 'EEG:F4-C4', 'EEG:C4-P4', 'EEG:P4-O2',
    'EEG:pad', 'EEG:I1', 'EEG:I2',
]

# HEAR: official upstream standard_1020 vocabulary used by HEAR spatial embeddings.
MODEL_CHANNEL_SPACE_HEAR = [
    'EEG:Fp1', 'EEG:Fpz', 'EEG:Fp2',
    'EEG:AF9', 'EEG:AF7', 'EEG:AF5', 'EEG:AF3', 'EEG:AF1', 'EEG:AFz', 'EEG:AF2', 'EEG:AF4', 'EEG:AF6', 'EEG:AF8', 'EEG:AF10',
    'EEG:F9', 'EEG:F7', 'EEG:F5', 'EEG:F3', 'EEG:F1', 'EEG:Fz', 'EEG:F2', 'EEG:F4', 'EEG:F6', 'EEG:F8', 'EEG:F10',
    'EEG:FT9', 'EEG:FT7', 'EEG:FC5', 'EEG:FC3', 'EEG:FC1', 'EEG:FCz', 'EEG:FC2', 'EEG:FC4', 'EEG:FC6', 'EEG:FT8', 'EEG:FT10',
    'EEG:T9', 'EEG:T7', 'EEG:C5', 'EEG:C3', 'EEG:C1', 'EEG:Cz', 'EEG:C2', 'EEG:C4', 'EEG:C6', 'EEG:T8', 'EEG:T10',
    'EEG:TP9', 'EEG:TP7', 'EEG:CP5', 'EEG:CP3', 'EEG:CP1', 'EEG:CPz', 'EEG:CP2', 'EEG:CP4', 'EEG:CP6', 'EEG:TP8', 'EEG:TP10',
    'EEG:P9', 'EEG:P7', 'EEG:P5', 'EEG:P3', 'EEG:P1', 'EEG:Pz', 'EEG:P2', 'EEG:P4', 'EEG:P6', 'EEG:P8', 'EEG:P10',
    'EEG:PO9', 'EEG:PO7', 'EEG:PO5', 'EEG:PO3', 'EEG:PO1', 'EEG:POz', 'EEG:PO2', 'EEG:PO4', 'EEG:PO6', 'EEG:PO8', 'EEG:PO10',
    'EEG:O1', 'EEG:Oz', 'EEG:O2', 'EEG:O9',
    'EEG:Iz', 'EEG:O10', 'EEG:T3', 'EEG:T5', 'EEG:T4', 'EEG:T6', 'EEG:M1', 'EEG:M2', 'EEG:A1', 'EEG:A2',
    'EEG:CCP1', 'EEG:CCP2', 'EEG:CCP3', 'EEG:CCP4', 'EEG:CCP5', 'EEG:CCP6',
    'EEG:T1', 'EEG:T2', 'EEG:FTT9h', 'EEG:TTP7h', 'EEG:TPP9h', 'EEG:FTT10h', 'EEG:TPP8h', 'EEG:TPP10h',
]

# NeuroRVQ EEG foundation model channel vocabulary (104 channels)
# Derived from the official NeuroRVQ EEG inference module channel list.
MODEL_CHANNEL_SPACE_NEURORVQ = [
    'EEG:A1', 'EEG:A2', 'EEG:AF3', 'EEG:AF4', 'EEG:AF7', 'EEG:AF8', 'EEG:AFz',
    'EEG:C1', 'EEG:C2', 'EEG:C3', 'EEG:C4', 'EEG:C5', 'EEG:C6',
    'EEG:CCP1', 'EEG:CCP2', 'EEG:CCP3', 'EEG:CCP4', 'EEG:CCP5', 'EEG:CCP6', 'EEG:CCP7', 'EEG:CCP8',
    'EEG:CFC1', 'EEG:CFC2', 'EEG:CFC3', 'EEG:CFC4', 'EEG:CFC5', 'EEG:CFC6', 'EEG:CFC7', 'EEG:CFC8',
    'EEG:CP1', 'EEG:CP2', 'EEG:CP3', 'EEG:CP4', 'EEG:CP5', 'EEG:CP6', 'EEG:CPz', 'EEG:Cz',
    'EEG:EOG',
    'EEG:F1', 'EEG:F10', 'EEG:F2', 'EEG:F3', 'EEG:F4', 'EEG:F5', 'EEG:F6', 'EEG:F7', 'EEG:F8', 'EEG:F9',
    'EEG:FC1', 'EEG:FC2', 'EEG:FC3', 'EEG:FC4', 'EEG:FC5', 'EEG:FC6', 'EEG:FCz',
    'EEG:Fp1', 'EEG:Fp2', 'EEG:Fpz', 'EEG:FT7', 'EEG:FT8', 'EEG:Fz',
    'EEG:Iz', 'EEG:LOC',
    'EEG:O1', 'EEG:O2', 'EEG:Oz',
    'EEG:P08', 'EEG:P1', 'EEG:P10', 'EEG:P2', 'EEG:P3', 'EEG:P4', 'EEG:P5', 'EEG:P6', 'EEG:P7', 'EEG:P8', 'EEG:P9',
    'EEG:PO1', 'EEG:PO10', 'EEG:PO2', 'EEG:PO3', 'EEG:PO4', 'EEG:PO7', 'EEG:PO8', 'EEG:PO9', 'EEG:POz',
    'EEG:Pz', 'EEG:ROC', 'EEG:SP1', 'EEG:SP2',
    'EEG:T1', 'EEG:T10', 'EEG:T2', 'EEG:T3', 'EEG:T4', 'EEG:T5', 'EEG:T6', 'EEG:T7', 'EEG:T8', 'EEG:T9',
    'EEG:TP10', 'EEG:TP7', 'EEG:TP8', 'EEG:TP9',
]

# FEMBA: TCP bipolar targets plus source electrode space used to construct them.
# The FEMBA backbone is pretrained on a TUH bipolar montage with channels such as
# FP1-F7, F7-T3, ..., A1-T3, T4-A2. We keep both already-bipolar names and the
# required single-electrode sources (including common aliases like T7/T8/P7/P8
# and M1/M2) so the wrapper can either pass through or reconstruct the montage.
MODEL_CHANNEL_SPACE_FEMBA = [
    'EEG:Fp1-F7', 'EEG:F7-T3', 'EEG:T3-T5', 'EEG:T5-O1',
    'EEG:Fp2-F8', 'EEG:F8-T4', 'EEG:T4-T6', 'EEG:T6-O2',
    'EEG:T3-C3', 'EEG:C3-Cz', 'EEG:Cz-C4', 'EEG:C4-T4',
    'EEG:Fp1-F3', 'EEG:F3-C3', 'EEG:C3-P3', 'EEG:P3-O1',
    'EEG:Fp2-F4', 'EEG:F4-C4', 'EEG:C4-P4', 'EEG:P4-O2',
    'EEG:A1-T3', 'EEG:T4-A2',
    'EEG:Fp1', 'EEG:Fp2',
    'EEG:F7', 'EEG:F8',
    'EEG:T3', 'EEG:T4', 'EEG:T5', 'EEG:T6',
    'EEG:T7', 'EEG:T8', 'EEG:P7', 'EEG:P8',
    'EEG:F3', 'EEG:F4',
    'EEG:C3', 'EEG:Cz', 'EEG:C4',
    'EEG:P3', 'EEG:P4',
    'EEG:O1', 'EEG:O2',
    'EEG:A1', 'EEG:A2', 'EEG:M1', 'EEG:M2',
]

# MIRepNet: 45-channel motor-imagery template from upstream `use_channels_names`
MODEL_CHANNEL_SPACE_MIREPNET = [
    'EEG:F7', 'EEG:F5', 'EEG:F3', 'EEG:F1', 'EEG:Fz', 'EEG:F2', 'EEG:F4', 'EEG:F6', 'EEG:F8',
    'EEG:FT7', 'EEG:FC5', 'EEG:FC3', 'EEG:FC1', 'EEG:FCz', 'EEG:FC2', 'EEG:FC4', 'EEG:FC6', 'EEG:FT8',
    'EEG:T7', 'EEG:C5', 'EEG:C3', 'EEG:C1', 'EEG:Cz', 'EEG:C2', 'EEG:C4', 'EEG:C6', 'EEG:T8',
    'EEG:TP7', 'EEG:CP5', 'EEG:CP3', 'EEG:CP1', 'EEG:CPz', 'EEG:CP2', 'EEG:CP4', 'EEG:CP6', 'EEG:TP8',
    'EEG:P7', 'EEG:P5', 'EEG:P3', 'EEG:P1', 'EEG:Pz', 'EEG:P2', 'EEG:P4', 'EEG:P6', 'EEG:P8',
]


def _canonical_channel_lookup_key(name: str) -> str:
    return name.replace('EEG:', '').replace(' ', '').strip().upper()


NEUROLM_CHANNEL_INDEX = {
    _canonical_channel_lookup_key(name): idx
    for idx, name in enumerate(MODEL_CHANNEL_SPACE_NEUROLM)
}

# Map model names to their channel spaces
MODEL_CHANNEL_SPACES: Dict[str, List[str]] = {
    'bendr': MODEL_CHANNEL_SPACE_BENDR,
    'eegpt': MODEL_CHANNEL_SPACE_EEGPT,
    'lead': MODEL_CHANNEL_SPACE_LEAD,
    'labram': MODEL_CHANNEL_SPACE_LABRAM,
    'hear': MODEL_CHANNEL_SPACE_HEAR,
    'mdjpt': MODEL_CHANNEL_SPACE_MDJPT,
    'neurogpt': MODEL_CHANNEL_SPACE_NEUROGPT,
    'neurolm': MODEL_CHANNEL_SPACE_NEUROLM,
    'neurorvq': MODEL_CHANNEL_SPACE_NEURORVQ,
    'femba': MODEL_CHANNEL_SPACE_FEMBA,
    'mirepnet': MODEL_CHANNEL_SPACE_MIREPNET,
    'steegformer': MODEL_CHANNEL_SPACE_STEEGFORMER,
}

BIOT_CHANNEL_SPACE_VARIANTS: Dict[str, List[str]] = {
    'prest-16ch': MODEL_CHANNEL_SPACE_BIOT_PREST_16CH,
    'shhs-prest-18ch': MODEL_CHANNEL_SPACE_BIOT_18CH,
    'six-datasets-18ch': MODEL_CHANNEL_SPACE_BIOT_18CH,
}
DEFAULT_BIOT_CHANNEL_SPACE_VARIANT = 'shhs-prest-18ch'


def get_channel_system(name: str = '10-20') -> List[str]:
    """
    Get channel system by name

    Args:
        name: Concrete channel system name. Public CLI choices are
            10-20/10-10/10-5/dynamic/custom, while internal dataset-specific systems
            are resolved before this function is called.

    Returns:
        List of channel names in the specified system

    Raises:
        KeyError: If channel system name is not found
    """
    if name not in CHANNEL_SYSTEMS:
        available = list(CHANNEL_SYSTEMS.keys())
        raise KeyError(
            f"Unknown channel system: '{name}'. "
            f"Available systems: {available}"
        )
    return CHANNEL_SYSTEMS[name]


def get_model_channel_space(model_name: str, variant: Optional[str] = None) -> Optional[List[str]]:
    """
    Get model-specific input channel space.
    Models not listed in MODEL_CHANNEL_SPACES return None, meaning the
    dataloader should respect the user-selected channel system directly.

    Args:
        model_name: Model name (e.g., 'eegpt', 'labram')
        variant: Optional pretrained-weight variant. Currently used by BIOT
            because its pretrained channel-token table is variant-specific.

    Returns:
        List of channel names in the model's input space, or None when the
        model does not declare a dedicated pretrained channel space.
    """
    model_key = model_name.lower()
    if model_key == 'biot':
        variant_key = (variant or DEFAULT_BIOT_CHANNEL_SPACE_VARIANT).lower()
        if variant_key not in BIOT_CHANNEL_SPACE_VARIANTS:
            available = list(BIOT_CHANNEL_SPACE_VARIANTS.keys())
            raise KeyError(
                f"Unknown BIOT channel-space variant: '{variant}'. "
                f"Available variants: {available}"
            )
        return BIOT_CHANNEL_SPACE_VARIANTS[variant_key]

    if model_key in MODEL_CHANNEL_SPACES:
        return MODEL_CHANNEL_SPACES[model_key]
    return None


def list_channel_systems() -> Dict[str, int]:
    """
    List all available channel systems and their channel counts

    Returns:
        Dict mapping system names to channel counts
    """
    return {name: len(channels) for name, channels in CHANNEL_SYSTEMS.items()}
