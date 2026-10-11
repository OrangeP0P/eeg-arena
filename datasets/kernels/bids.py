"""BIDS stage1 kernel for EEGUnity.

Extracts participant metadata (age, sex) and event summaries from BIDS-formatted
EEG datasets. Reads participants.tsv and *_events.tsv without modifying raw EEG
data or channel configuration. Falls back to raw.annotations when no events.tsv
is available.

Works with preload=False since no raw data access is needed (annotations are
loaded from the file header even without preloading).

Spec example
------------
"./kernels/bids.py"
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Tuple

import mne
import pandas as pd


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _find_participants_tsv(file_path: str) -> Optional[str]:
    """Walk up the directory tree to find the nearest participants.tsv."""
    current = os.path.dirname(os.path.abspath(file_path))
    prev = None
    while current != prev:
        candidate = os.path.join(current, "participants.tsv")
        if os.path.isfile(candidate):
            return candidate
        prev = current
        current = os.path.dirname(current)
    return None


def _find_events_tsv(file_path: str) -> Optional[str]:
    """Derive the BIDS events.tsv path from an EEG file path.

    Strategy (tried in order):
    1. File-level sidecar in the same directory:
       sub-001_task-PES_eeg.edf  →  sub-001_task-PES_events.tsv
    2. Dataset-level sidecar at the participants.tsv directory:
       task-PES_events.tsv  (shared across all subjects)
    """
    dirname = os.path.dirname(file_path)
    basename = os.path.basename(file_path)
    name_no_ext = os.path.splitext(basename)[0]
    name_base = re.sub(r'_[^_]+$', '', name_no_ext)

    # 1. File-level sidecar (same directory)
    candidate = os.path.join(dirname, name_base + "_events.tsv")
    if os.path.isfile(candidate):
        return candidate

    # 2. Dataset-level sidecar (participants.tsv directory)
    tsv_root = _find_participants_tsv(file_path)
    if tsv_root:
        dataset_root = os.path.dirname(tsv_root)
        m = re.search(r'task-([^_]+)', basename)
        if m:
            ds_events = os.path.join(dataset_root, f"task-{m.group(1)}_events.tsv")
            if os.path.isfile(ds_events):
                return ds_events

    return None


def _load_sex_levels(tsv_path: str) -> dict:
    """Read participants.json alongside participants.tsv to decode numeric sex values.

    BIDS spec allows a sidecar JSON with a ``Levels`` dict, e.g.::

        {"sex": {"Levels": {"1": "male", "2": "female"}}}
        {"sex": {"Levels": {"one": "male", "two": "female"}}}

    Returns a dict mapping raw TSV values (strings) to normalised sex strings,
    e.g. ``{"1": "Male", "2": "Female"}``.  Empty dict if no usable JSON found.
    """
    json_path = os.path.splitext(tsv_path)[0] + '.json'
    if not os.path.isfile(json_path):
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        field = (meta.get('sex') or meta.get('Sex')
                 or meta.get('gender') or meta.get('Gender') or {})
        levels = field.get('Levels', {}) if isinstance(field, dict) else {}
        if not levels:
            return {}

        _WORD_TO_INT = {'zero': '0', 'one': '1', 'two': '2', 'three': '3',
                        'four': '4', 'five': '5'}
        result = {}
        for k, v in levels.items():
            normalised = _normalize_sex(str(v))
            k_str = str(k)
            result[k_str] = normalised
            # Register numeric equivalent when key is an English word ("one"→"1")
            k_lower = k_str.lower()
            if k_lower in _WORD_TO_INT:
                result[_WORD_TO_INT[k_lower]] = normalised
            # Register integer-stripped form ("1.0"→"1")
            try:
                result[str(int(float(k_str)))] = normalised
            except (ValueError, TypeError):
                pass
        return result
    except Exception:
        return {}


def _normalize_sex(val: str) -> str:
    """Normalize sex/gender value to a consistent string."""
    v = str(val).strip().lower()
    if v in ("f", "female"):
        return "Female"
    if v in ("m", "male"):
        return "Male"
    if v == "woman":
        return "Female"
    if v == "man":
        return "Male"
    if v == "w":
        return "Female"   # confirmed: ds004256 uses w=woman
    if v == "o":
        return "Other"    # abbreviated Other (ds007172)
    if v == "nb":
        return "Non-binary"
    if v in ("cannot", "unable", "not available"):
        return "unknown"
    if v in ("n/a", "na", "nan", "none", ""):
        return "unknown"
    return str(val).strip()


def _get_col_case_insensitive(df: pd.DataFrame, name: str) -> Optional[str]:
    """Return the actual column name matching `name` case-insensitively, or None."""
    name_lower = name.lower()
    for col in df.columns:
        if col.lower() == name_lower:
            return col
    return None


def _get_col_first_word(df: pd.DataFrame, name: str) -> Optional[str]:
    """Match a column whose *first whitespace-separated word* equals `name` (case-insensitive).

    Handles malformed TSV headers like ``sex education_level`` where two fields
    were accidentally merged into one column name.
    """
    name_lower = name.lower()
    for col in df.columns:
        if col.split()[0].lower() == name_lower:
            return col
    return None


# Generic BIDS modality/structure directory names that are never subject IDs.
_GENERIC_DIRS = frozenset(
    {"eeg", "ieeg", "meg", "mri", "fmri", "beh", "pet",
     "sourcedata", "derivatives", "code", "stimuli", ""}
)

# Annotation labels that are system/infrastructure markers, not task events.
_SYSTEM_ANNOTATIONS = frozenset(
    {"boundary", "New Segment/", "New Segment", "EDGE 99", ""}
)
_SYSTEM_ANNOTATION_PREFIXES = ("SyncStatus",)

# Ordered list of column names to try for event labels in events.tsv.
_EVENT_LABEL_COLS = (
    "trial_type", "value", "type", "stim_type",
    "condition", "stimulus", "event_type",
)

# Maximum character length for the events summary string stored in the locator.
# HED-annotated datasets (e.g. ds004460) can produce values exceeding 900 KB per
# row.  Values exceeding this limit are truncated with a marker.
_MAX_EVENTS_LEN = 2000


def _extract_subject_id(file_path: str) -> str:
    """Extract the BIDS subject ID from an EEG file path robustly.

    Strategy (tried in order):
    1. Regex on filename — BIDS requires the filename to start with ``sub-<label>_``.
       Handles any nesting depth (sub/eeg/, sub/ses/eeg/, derivatives/.../sub/eeg/).
    2. Scan path components from deepest to shallowest for a ``sub-*`` directory.
    3. Fallback: first parent directory that is not a known generic BIDS directory name
       (handles non-BIDS layouts like sourcedata/S01/file.bdf).
    """
    basename = os.path.basename(file_path)

    # 1. Extract sub-<label> directly from filename (most reliable for BIDS)
    m = re.match(r'(sub-[^_]+)', basename)
    if m:
        return m.group(1)

    # 2. Walk path components looking for a sub-* directory
    parts = os.path.abspath(file_path).split(os.sep)
    for part in reversed(parts[:-1]):   # skip the filename itself
        if re.match(r'sub-', part, re.IGNORECASE):
            return part

    # 3. Walk parent directories, return the first non-generic one
    current = os.path.dirname(os.path.abspath(file_path))
    prev = None
    while current != prev:
        part = os.path.basename(current)
        if part.lower() not in _GENERIC_DIRS:
            return part
        prev = current
        current = os.path.dirname(current)

    # Ultimate fallback
    return os.path.basename(os.path.dirname(os.path.dirname(file_path)))


def _is_system_annotation(label: str) -> bool:
    """Return True if the annotation label is a system/infrastructure marker."""
    if label in _SYSTEM_ANNOTATIONS:
        return True
    for prefix in _SYSTEM_ANNOTATION_PREFIXES:
        if label.startswith(prefix):
            return True
    return False


# ---------------------------------------------------------------------------
# Kernel class
# ---------------------------------------------------------------------------

@dataclass
class BIDSKernel:
    """Stage1 kernel for BIDS-formatted EEG datasets.

    Reads age, sex, and event summary from BIDS sidecar files and stores
    them in raw.info['description'] under the 'eegunity_description' key.
    Falls back to raw.annotations for event extraction when no events.tsv
    is available or usable.
    """

    KERNEL_ID: str = "bids-stage1"
    _participants_cache: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Public helpers (can be called without a raw object)
    # ------------------------------------------------------------------

    def get_age_sex(self, file_path: str) -> Tuple[str, str]:
        """Extract age and sex for the subject associated with file_path.

        Looks for participants.tsv by walking up the directory tree.
        Handles column name variants (sex/gender, case-insensitive) and
        merged column names (e.g. ``sex education_level``).

        Subject ID matching tries multiple strategies:
        - Direct match
        - Hyphen/underscore swap  (e.g. ``sub-01`` ↔ ``sub_01``)
        - Numeric suffix match  (e.g. ``sub-FFE001`` → ``sub-001``)

        Returns
        -------
        (age, sex) — strings; "unknown" when not available.
        """
        tsv_path = _find_participants_tsv(file_path)
        if tsv_path is None:
            return "unknown", "unknown"

        if tsv_path not in self._participants_cache:
            try:
                self._participants_cache[tsv_path] = pd.read_csv(tsv_path, sep="\t")
            except Exception:
                self._participants_cache[tsv_path] = None

        df = self._participants_cache[tsv_path]
        if df is None:
            return "unknown", "unknown"

        pid_col = _get_col_case_insensitive(df, "participant_id")
        if pid_col is None:
            return "unknown", "unknown"

        subject_id = _extract_subject_id(file_path)
        match = self._match_subject(df, pid_col, subject_id)
        if match.empty:
            return "unknown", "unknown"

        age = self._extract_age(match, df)
        sex_levels = _load_sex_levels(tsv_path)
        sex = self._extract_sex(match, df, sex_levels=sex_levels)
        return age, sex

    def _match_subject(
        self, df: pd.DataFrame, pid_col: str, subject_id: str
    ) -> pd.DataFrame:
        """Try multiple subject ID variants to find a row in participants.tsv."""
        # 1. Direct match
        match = df[df[pid_col] == subject_id]
        if not match.empty:
            return match

        # 2. Hyphen/underscore swap (handles sub-01 vs sub_01)
        if '-' in subject_id:
            alt = subject_id.replace('-', '_')
        else:
            alt = subject_id.replace('_', '-')
        match = df[df[pid_col] == alt]
        if not match.empty:
            return match

        # 3. Numeric suffix match (handles sub-FFE001 vs sub-001)
        m = re.search(r'(\d+)$', subject_id)
        if m:
            numeric = m.group(1).lstrip('0') or '0'
            for pid in df[pid_col].dropna().astype(str):
                m2 = re.search(r'(\d+)$', pid)
                if m2 and (m2.group(1).lstrip('0') or '0') == numeric:
                    match = df[df[pid_col] == pid]
                    if not match.empty:
                        return match

        return pd.DataFrame()

    def _extract_age(self, match: pd.DataFrame, df: pd.DataFrame) -> str:
        """Extract age value from a matched participants row.

        Accepts numeric values and BIDS age-range strings (e.g. '18-38').
        Returns 'unknown' for non-numeric strings (e.g. 'F', 'M') that
        indicate the age/sex columns are swapped in the source TSV.
        """
        age_col = _get_col_case_insensitive(df, "age")
        if age_col is None:
            return "unknown"
        val = match[age_col].values[0]
        if not pd.notna(val):
            return "unknown"
        val_str = str(val).strip()
        if val_str.lower() in ("n/a", "na", ""):
            return "unknown"
        # Validate: must be numeric or an age-range like "18-38"
        try:
            float(val_str)
            return val_str
        except ValueError:
            if re.match(r'^\d+[\-\u2013]\d+$', val_str):  # e.g. 18-38 or 18–38
                return val_str
            return "unknown"

    def _extract_sex(self, match: pd.DataFrame, df: pd.DataFrame,
                     sex_levels: dict = None) -> str:
        """Extract sex/gender value from a matched participants row.

        Tries exact column name match first, then first-word match for
        merged column names like ``sex education_level``.

        Parameters
        ----------
        sex_levels : dict, optional
            Mapping from raw TSV value strings to normalised sex strings, as
            returned by ``_load_sex_levels()``.  When provided and the raw
            value is found in the mapping it takes precedence over
            ``_normalize_sex()``.
        """
        # Exact (case-insensitive) match: prefer 'sex' over 'gender'
        sex_col = (
            _get_col_case_insensitive(df, "sex")
            or _get_col_case_insensitive(df, "gender")
        )
        # Fallback: first-word match for merged columns
        if sex_col is None:
            sex_col = (
                _get_col_first_word(df, "sex")
                or _get_col_first_word(df, "gender")
            )
        if sex_col is None:
            return "unknown"

        val = match[sex_col].values[0]
        if not pd.notna(val):
            return "unknown"

        raw_val = str(val)
        # For merged column values like "f   1", take the first token
        if ' ' in raw_val.strip():
            raw_val = raw_val.split()[0]

        # Try BIDS participants.json Levels mapping first
        if sex_levels:
            if raw_val in sex_levels:
                return sex_levels[raw_val]
            # Try integer-stripped form ("1.0" → "1")
            try:
                int_key = str(int(float(raw_val)))
                if int_key in sex_levels:
                    return sex_levels[int_key]
            except (ValueError, TypeError):
                pass

        normalized = _normalize_sex(raw_val)
        # Guard: if the result is a large number it is almost certainly an age
        # value in a dataset where the sex/age columns are swapped (e.g. ds004295,
        # ds005383).  Treat as unknown rather than storing a nonsensical value.
        try:
            if float(normalized) > 10:
                return "unknown"
        except (ValueError, TypeError):
            pass
        return normalized

    def get_events_summary(self, file_path: str) -> Optional[str]:
        """Build an event-count summary string from a BIDS *_events.tsv sidecar.

        Tries column names in order: trial_type, value, type, stim_type,
        condition, stimulus, event_type.

        Returns
        -------
        str
            A count string like "rest: 10, left_hand: 50" if data was found,
            ``"no_events"`` if the file was found but has only n/a values,
            or ``None`` if no usable events.tsv was found (caller should try
            raw.annotations next).
        """
        events_path = _find_events_tsv(file_path)
        if events_path is None:
            return None

        try:
            df = pd.read_csv(events_path, sep="\t")
        except Exception:
            return None

        label_col = None
        for candidate in _EVENT_LABEL_COLS:
            if candidate in df.columns:
                label_col = candidate
                break

        if label_col is None:
            return None  # no recognised label column → try annotations

        counts = df[label_col].dropna().value_counts()
        # Filter out BIDS "n/a" placeholder strings
        counts = counts[~counts.index.astype(str).str.lower().isin({"n/a", "na", ""})]
        if counts.empty:
            return "no_events"

        return ", ".join(
            f"{str(label).replace(chr(34), chr(39))}: {cnt}" for label, cnt in counts.items()
        )

    def get_events_from_annotations(self, raw: mne.io.BaseRaw) -> str:
        """Build an event-count summary from raw.annotations.

        Skips system/infrastructure labels (boundary, New Segment/, SyncStatus/*).

        Returns
        -------
        str
            A count string, or ``"no_events"`` if no task annotations exist.
        """
        ann = raw.annotations
        if len(ann) == 0:
            return "no_events"

        counts = Counter(ann.description)
        task_counts = {
            label: cnt
            for label, cnt in counts.items()
            if not _is_system_annotation(label)
        }
        if not task_counts:
            return "no_events"

        sorted_counts = sorted(task_counts.items(), key=lambda x: -x[1])
        return ", ".join(
            f"{str(label).replace(chr(34), chr(39))}: {cnt}" for label, cnt in sorted_counts
        )

    # ------------------------------------------------------------------
    # EEGUnity kernel interface
    # ------------------------------------------------------------------

    def apply(self, udataset, raw: mne.io.BaseRaw, row) -> mne.io.BaseRaw:
        """Apply BIDS metadata extraction to a single record.

        Parameters
        ----------
        udataset
            EEGUnity EEGParser context (shared-attr holder).
        raw
            In-memory MNE Raw object (may be loaded with preload=False).
            Annotations are available even without full data preloading.
        row
            Locator row (pandas.Series-like) with at least 'File Path'.

        Returns
        -------
        mne.io.BaseRaw
            The raw object with updated info['description'].
        """
        file_path = row["File Path"]

        age, sex = self.get_age_sex(file_path)

        # Events: try events.tsv first; fall back to raw.annotations
        events_summary = self.get_events_summary(file_path)
        if events_summary is None:
            events_summary = self.get_events_from_annotations(raw)

        # Truncate oversized events (e.g. HED tags can exceed 900 KB per row)
        if events_summary and len(events_summary) > _MAX_EVENTS_LEN:
            events_summary = events_summary[:_MAX_EVENTS_LEN] + "...[truncated]"

        description_dict = {
            "original_description": raw.info.get("description", ""),
            "eegunity_description": {
                "age": age,
                "sex": sex,
                "events": events_summary,
            },
        }
        raw.info["description"] = json.dumps(description_dict)
        return raw


KERNEL = BIDSKernel()
