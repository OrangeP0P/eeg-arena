"""OpenNeuro DS001784 stage1 kernel for EEGUnity.

Effects of ON/OFF deep brain stimulation on cognitive control in treatment-resistant
depression (EEG). BIDS-formatted dataset with 65 channels at 1450 Hz.

Dataset: openneuro_ds001784 | Year: 2019 | DOI: https://doi.org/10.1038/s41467-019-09557-4
Paradigm: MSIT with emotional stimuli + DBS ON/OFF

This kernel:
- Reads participants.tsv for age/sex/dbs metadata
- Parses events.tsv to extract trial_type (interference/control) labels
- Maps raw events (event_2/4/6) to semantic labels
- Sets standard 10-20 montage for 65-channel EEG
- Provides DBS ON/OFF labels from participants.tsv

Source: https://openneuro.org/datasets/ds001784
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mne
import pandas as pd


# =============================================================================
# Device Metadata Defaults
# =============================================================================
# Annotator: LLM (gpt-4o), 2026-04-08
# Source: Dataset publication DOI 10.1038/s41467-019-09557-4, methods section
# Amplifier, cap, and handedness cannot be confirmed from available documentation.
# These values are LLM inferences and require human verification.
_DEVICE_METADATA_DEFAULTS: Dict[str, str] = {
    "amplifier": "unknown",    # LLM 推断，需人类复核
    "cap": "unknown",          # LLM 推断，需人类复核
    "handedness": "unknown",   # LLM 推断，需人类复核
}


# =============================================================================
# Raw Event Mapping
# =============================================================================
# Annotator: LLM (gpt-4o), 2026-04-08
# Source: Dataset events.tsv analysis
# event_2 (1296 occurrences) -> interference trials (high-interference MSIT)
# event_4 (2592 occurrences) -> control trials (low interference)
# event_6 (1296 occurrences) -> AFMSIT trials (active foreground MSIT)
_EVENT_RAW_MAP: Dict[str, str] = {
    "event_2": "interference_trial",
    "event_4": "control_trial",
    "event_6": "afmsit_trial",
}


# =============================================================================
# Label Mapping for Super Meta
# =============================================================================
# Annotator: LLM (gpt-4o), 2026-04-08
# Source: analysis.md Classical Task 1 definition
# interference_trial -> y=1, control_trial -> y=0
_COGNITIVE_LABEL_MAP: Dict[str, int] = {
    "interference_trial": 1,
    "control_trial": 0,
}


# =============================================================================
# Standard 10-20 Channel Names for 64-65 Channel Systems
# =============================================================================
# Annotator: LLM (gpt-4o), 2026-04-08
# Source: Standard 10-20 system for 65-channel EEG (extended 10-20)
# These represent the standard electrode positions in order
_STANDARD_1020_CHANNELS: List[str] = [
    "Fp1", "Fpz", "Fp2",
    "AF9", "AF7", "AF5", "AF3", "AFz", "AF4", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "POz", "PO4", "PO8", "PO10",
    "O1", "Oz", "O2",
    "I1", "Iz", "I2",
]


# =============================================================================
# Helper Functions
# =============================================================================

def _load_participants_tsv(dataset_root: str) -> pd.DataFrame:
    """Load participants.tsv if exists."""
    pfile = os.path.join(dataset_root, 'participants.tsv')
    if os.path.exists(pfile):
        return pd.read_csv(pfile, sep='\t')
    return pd.DataFrame()


def _load_events_tsv(eeg_file_path: str) -> pd.DataFrame:
    """Load corresponding *_events.tsv file."""
    base, ext = os.path.splitext(eeg_file_path)
    events_path = base + '_events.tsv'
    if os.path.exists(events_path):
        return pd.read_csv(events_path, sep='\t')
    return pd.DataFrame()


def _get_subject_metadata(
    subject_id: str,
    participants_df: pd.DataFrame,
) -> Dict[str, str]:
    """Extract subject metadata from participants.tsv.
    
    Returns a dict with keys: age, sex, handedness, dbs_condition.
    """
    metadata = {
        "age": "unknown",
        "sex": "unknown",
        "handedness": "unknown",
        "dbs_condition": "unknown",
    }
    
    if participants_df.empty:
        return metadata
    
    # Normalize subject_id for matching
    normalized_id = subject_id.replace('sub-', '')
    
    # Try exact match in participant_id column
    if 'participant_id' in participants_df.columns:
        for idx, row in participants_df.iterrows():
            pid = str(row['participant_id']).replace('sub-', '')
            if pid == normalized_id:
                # Extract age
                if 'age' in row and pd.notna(row['age']):
                    metadata['age'] = str(int(row['age']))
                
                # Extract sex
                if 'sex' in row and pd.notna(row['sex']):
                    sex_val = str(row['sex']).strip().lower()
                    if sex_val in ['m', 'male']:
                        metadata['sex'] = 'male'
                    elif sex_val in ['f', 'female']:
                        metadata['sex'] = 'female'
                    else:
                        metadata['sex'] = 'unknown'
                
                # Extract handedness
                if 'handedness' in row and pd.notna(row['handedness']):
                    handed_val = str(row['handedness']).strip().lower()
                    if 'right' in handed_val:
                        metadata['handedness'] = 'right'
                    elif 'left' in handed_val:
                        metadata['handedness'] = 'left'
                    elif 'ambi' in handed_val:
                        metadata['handedness'] = 'ambidextrous'
                
                # Extract DBS condition
                if 'dbs_condition' in row and pd.notna(row['dbs_condition']):
                    dbs_val = str(row['dbs_condition']).strip().lower()
                    if dbs_val in ['on', '1', 'dbs_on']:
                        metadata['dbs_condition'] = 'DBS_ON'
                    elif dbs_val in ['off', '0', 'dbs_off']:
                        metadata['dbs_condition'] = 'DBS_OFF'
                
                break
    
    return metadata


def _build_channel_rename_map(raw: mne.io.Raw) -> Dict[str, str]:
    """Build a rename map to standardize channel names.
    
    Attempts to map non-standard names (EEG-0, EEG 001, etc.) to 
    standard 10-20 electrode names based on channel count and position.
    """
    rename_map: Dict[str, str] = {}
    existing_names = set(raw.ch_names)
    
    # Count EEG channels
    eeg_channels = [ch for ch in raw.ch_names 
                   if ch.startswith('EEG-') or ch.startswith('EEG ')]
    
    if len(eeg_channels) >= 60:
        # Sort EEG channels by their numeric suffix
        def extract_num(name: str) -> int:
            # Extract trailing number from EEG-0, EEG 001, etc.
            import re
            match = re.search(r'(\d+)$', name)
            return int(match.group(1)) if match else 0
        
        eeg_channels_sorted = sorted(eeg_channels, key=extract_num)
        
        # Standard 64-channel layout (excluding reference)
        # Based on extended 10-20 system
        standard_64 = [
            "Fp1", "Fpz", "Fp2",
            "F7", "F3", "Fz", "F4", "F8",
            "FC5", "FC1", "FC2", "FC6",
            "T7", "C3", "Cz", "C4", "T8",
            "CP5", "CP1", "CP2", "CP6",
            "P7", "P3", "Pz", "P4", "P8",
            "PO7", "PO3", "POz", "PO4", "PO8",
            "O1", "Oz", "O2",
            # Additional 32 channels for 64+ system
            "AF7", "AF3", "AF4", "AF8",
            "F5", "F1", "F2", "F6",
            "FC3", "FCz", "FC4",
            "C5", "C1", "C2", "C6",
            "CP3", "CPz", "CP4",
            "P5", "P1", "P2", "P6",
            "PO5", "PO6",
            "TP7", "TP8",
            "PO9", "PO10",
            "AFz", "F2", "F10",
        ]
        
        # Map each EEG channel to standard name
        for i, eeg_ch in enumerate(eeg_channels_sorted):
            if i < len(standard_64) and eeg_ch != standard_64[i]:
                rename_map[eeg_ch] = standard_64[i]
    
    # Handle common naming variants
    for ch in existing_names:
        # Already standard names - no change needed
        if ch in _STANDARD_1020_CHANNELS:
            continue
        # Common variant patterns
        if ch.startswith('EEG-') or ch.startswith('EEG '):
            continue  # Will be handled by numeric mapping above
    
    return rename_map


def _create_annotations_from_events(
    raw: mne.io.Raw,
    events_df: pd.DataFrame,
) -> mne.Annotations:
    """Create MNE annotations from events.tsv data.
    
    Maps event IDs to semantic labels based on _EVENT_RAW_MAP.
    """
    if events_df.empty:
        return raw.annotations
    
    annotations_list = []
    onset_list = []
    duration_list = []
    
    # Process events based on available columns
    # Standard BIDS events.tsv columns: onset, duration, trial_type, value
    
    for _, event_row in events_df.iterrows():
        onset = float(event_row.get('onset', 0))
        duration = float(event_row.get('duration', 0))
        
        # Try to get event value/description
        event_value = str(event_row.get('value', ''))
        event_id = str(event_row.get('event_id', event_value))
        
        # Map to semantic label
        if event_id in _EVENT_RAW_MAP:
            label = _EVENT_RAW_MAP[event_id]
        else:
            label = f"event_{event_id}"
        
        # Get trial_type if available for richer labels
        trial_type = event_row.get('trial_type', None)
        if pd.notna(trial_type):
            # Use trial_type as primary label if available
            label = str(trial_type)
            # Also keep original event_id mapping for raw meta
            orig_label = f"{event_id}:{label}"
        else:
            orig_label = label
        
        onset_list.append(onset)
        duration_list.append(duration)
        annotations_list.append(orig_label)
    
    return mne.Annotations(
        onset=onset_list,
        duration=duration_list,
        description=annotations_list,
    )


def _build_eegunity_description(
    original_desc: str,
    age: str,
    sex: str,
    amplifier: str,
    cap: str,
    handedness: str,
) -> str:
    """Build the eegunity_description JSON string."""
    description_dict = {
        "original_description": original_desc,
        "eegunity_description": {
            "age": age,
            "sex": sex,
            "amplifier": amplifier,
            "cap": cap,
            "handedness": handedness,
        },
    }
    return json.dumps(description_dict, ensure_ascii=False)


# =============================================================================
# Kernel Class
# =============================================================================

@dataclass
class OpenNeuroDS001784Kernel:
    """Stage1 kernel for OpenNeuro DS001784 dataset.
    
    Effects of ON/OFF deep brain stimulation on cognitive control 
    in treatment-resistant depression (EEG).
    """
    
    KERNEL_ID: str = "openneuro-ds001784-stage1"
    
    def apply(
        self,
        raw: mne.io.Raw,
        row: Optional[Any] = None,
        **kwargs,
    ) -> mne.io.Raw:
        """Apply stage1 preprocessing to a single EEG record.
        
        Parameters
        ----------
        raw
            In-memory MNE Raw object.
        row
            Locator row (pandas.Series-like) with at least:
            - File Path or file_path
            - subject_id or Subject ID
            - Dataset Root or dataset_root
            Defaults to None if not provided.
        **kwargs
            Additional keyword arguments for forward compatibility.
        
        Returns
        -------
        mne.io.Raw
            The modified Raw object with updated annotations and info.
        """
        # Extract file path and dataset root
        file_path = ''
        dataset_root = ''
        subject_id = 'unknown'
        
        if row is not None:
            file_path = row.get('File Path', row.get('file_path', ''))
            dataset_root = row.get('Dataset Root', row.get('dataset_root', ''))
            subject_id = row.get('Subject ID', row.get('subject_id', 'unknown'))
        
        # Normalize subject_id
        if subject_id.startswith('sub-'):
            subject_id_clean = subject_id
        else:
            subject_id_clean = f"sub-{subject_id}"
        
        # Load participants.tsv for metadata
        participants_df = _load_participants_tsv(dataset_root)
        
        # Get subject metadata
        subject_meta = _get_subject_metadata(subject_id_clean, participants_df)
        
        # Load events.tsv
        events_df = _load_events_tsv(file_path)
        
        # Store DBS condition as annotation for later use
        dbs_condition = subject_meta.get('dbs_condition', 'unknown')
        
        # Create annotations from events
        if not events_df.empty:
            # Check for trial_type column
            if 'trial_type' in events_df.columns:
                # Map trial_type to semantic labels with event prefix
                annotations_list = []
                onset_list = []
                duration_list = []
                
                for _, event_row in events_df.iterrows():
                    onset = float(event_row.get('onset', 0))
                    duration = float(event_row.get('duration', 0))
                    trial_type = str(event_row.get('trial_type', ''))
                    event_id = str(event_row.get('value', event_row.get('event_id', '')))
                    
                    # Create label: event_id:trial_type
                    label = f"{event_id}:{trial_type}"
                    
                    onset_list.append(onset)
                    duration_list.append(duration)
                    annotations_list.append(label)
                
                annotations = mne.Annotations(
                    onset=onset_list,
                    duration=duration_list,
                    description=annotations_list,
                )
            else:
                # Use raw event mapping
                annotations = _create_annotations_from_events(raw, events_df)
        else:
            # No events.tsv found, keep existing annotations
            annotations = raw.annotations
        
        raw.set_annotations(annotations)
        
        # Channel renaming (must happen BEFORE set_montage)
        rename_map = _build_channel_rename_map(raw)
        if rename_map:
            # Check which channels actually exist in raw
            actual_renames = {k: v for k, v in rename_map.items() if k in raw.ch_names}
            if actual_renames:
                print(f"[OpenNeuroDS001784] Renaming channels: {actual_renames}")
                raw.rename_channels(actual_renames)
        
        # Set standard 10-20 montage if channels match
        try:
            # Get a standard montage for 65 channels
            # Try standard_1020 first
            montage = mne.channels.make_standard_montage('standard_1020')
            
            # Get existing channel names that are in the montage
            existing_eeg = [ch for ch in raw.ch_names 
                           if ch in montage.ch_names and 
                           not ch.startswith('EOG') and 
                           not ch.startswith('EMG') and
                           not ch.startswith('ECG')]
            
            if len(existing_eeg) >= 30:
                # Pick a subset of montage channels that exist in data
                available_montage_chs = [ch for ch in montage.ch_names 
                                        if ch in existing_eeg]
                if available_montage_chs:
                    raw.set_montage(montage, on_missing='ignore')
                    print(f"[OpenNeuroDS001784] Set standard_1020 montage for {len(available_montage_chs)} channels")
        except Exception as e:
            print(f"[OpenNeuroDS001784] Could not set montage: {e}")
        
        # Build eegunity_description
        original_desc = raw.info.get('description', '')
        age = subject_meta.get('age', 'unknown')
        sex = subject_meta.get('sex', 'unknown')
        amplifier = _DEVICE_METADATA_DEFAULTS['amplifier']
        cap = _DEVICE_METADATA_DEFAULTS['cap']
        handedness = subject_meta.get('handedness', 'unknown')
        
        description_json = _build_eegunity_description(
            original_desc=original_desc,
            age=age,
            sex=sex,
            amplifier=amplifier,
            cap=cap,
            handedness=handedness,
        )
        
        raw.info['description'] = description_json
        
        # Store DBS condition for super meta access
        # EEGUnity can access this via raw.annotations.description
        if dbs_condition != 'unknown':
            # Add DBS label to raw.info for downstream access
            raw.info['dbs_condition'] = dbs_condition
        
        return raw


# =============================================================================
# Module-level Kernel Instance
# =============================================================================

KERNEL = OpenNeuroDS001784Kernel()