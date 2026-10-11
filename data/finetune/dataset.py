import json
import os
import re
import h5py
import torch
from typing import List, Dict, Optional
import numpy as np
from torch.utils.data import Sampler, Subset

from datasets import get_dataset_config


def _parse_subject_id(groupname: str, parser_cfg: Dict) -> str:
    """Parse subject id from group name using registry-provided parser config."""
    parser_type = parser_cfg.get("type", "slice")

    if parser_type == "slice":
        start = parser_cfg.get("start", 0)
        end = parser_cfg.get("end", None)
        subject_id = groupname[slice(start, end)]
        return subject_id if subject_id else groupname

    if parser_type == "split":
        sep = parser_cfg.get("sep", "_")
        index = int(parser_cfg.get("index", 0))
        parts = groupname.split(sep)
        if not parts:
            return groupname
        if -len(parts) <= index < len(parts):
            subject_id = parts[index]
            return subject_id if subject_id else groupname
        return parts[0]

    if parser_type == "regex":
        pattern = parser_cfg.get("pattern")
        if not pattern:
            return groupname
        match = re.search(pattern, groupname)
        if match is None:
            return groupname
        group = parser_cfg.get("group", 0)
        try:
            subject_id = match.group(group)
        except IndexError:
            subject_id = match.group(0)
        return subject_id if subject_id else groupname

    raise ValueError(f"Unsupported subject_parser type: {parser_type}")


def get_subject_by_groupname(groupname: str, dataset_name: str) -> str:
    ds_config = get_dataset_config(dataset_name)
    if ds_config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    parser_cfg = ds_config.get("subject_parser")
    if parser_cfg is None:
        return groupname

    return _parse_subject_id(groupname, parser_cfg)


class EpochDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path: str, events: Dict[str, int], channels_select: Optional[List[str]] = None,
                 extra_info: Optional[List[str]] = None, dataset_name: str = 'unspecified',
                 pad_to_multiple: Optional[int] = None):
        if not isinstance(events, dict):
            raise ValueError(
                "Parameter 'events' must be a dictionary with event names as keys and integer labels as values.")

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found at the specified path: {h5_path}")

        self.h5_path = h5_path
        self.dataset_name = dataset_name
        self.events = events
        self.channels_select = channels_select
        self.extra_info = extra_info if extra_info is not None else []
        self.pad_to_multiple = pad_to_multiple
        self.index = []
        self._h5_file = None
        self._ch_cache = {}  # Cache channel indices
        self._selected_flat_indices = []
        self._epoch_meta_cache = {}
        self._available_epoch_meta_fields = set()

        print(f"Loading HDF5 data from {h5_path}")
        self._build_index()

        self.total_epochs = len(self.index)

    def __len__(self):
        return self.total_epochs

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise IndexError("Index out of range")

        group_name, flat_idx, label = self.index[idx]

        f = self._get_h5_file()
        epoch_data = f['data'][flat_idx]  # (n_ch, n_times)

        if self.channels_select:
            ch_idx = self._ch_cache[None]
            epoch_data = epoch_data[ch_idx, :]

        # Pad to be divisible by pad_to_multiple (if specified)
        if self.pad_to_multiple is not None:
            time_len = epoch_data.shape[-1]
            pad_len = (self.pad_to_multiple - time_len % self.pad_to_multiple) % self.pad_to_multiple
            if pad_len > 0:
                padding = [(0, 0)] * (epoch_data.ndim - 1) + [(0, pad_len)]
                epoch_data = np.pad(epoch_data, padding, mode='constant', constant_values=0)

        # Channel info from root attrs (v2: global for all epochs)
        ch_info = {}
        ch_info['chOrder'] = np.array(json.loads(f.attrs['ch_names']))
        for field_name in self.extra_info:
            if field_name == 'chOrder':
                continue
            if self.has_epoch_meta(field_name):
                ch_info[field_name] = self._get_epoch_meta_value(field_name, idx)

        data = torch.tensor(epoch_data, dtype=torch.float32)
        subject = get_subject_by_groupname(group_name, self.dataset_name)

        return data, label, ch_info, subject

    def _build_index(self):
        # Build index in the main process, but do not keep a long-lived shared HDF5 handle.
        with h5py.File(self.h5_path, 'r') as h5_file:
            version = h5_file.attrs.get('version', None)
            if version != '2.0':
                raise ValueError(
                    f"HDF5 file '{self.h5_path}' is v1 format (version={version!r}), "
                    "which is no longer supported. Delete the cached file and re-run "
                    "with --rebuild_hdf5 to generate a v2 file."
                )

            # label_map: {int_code -> event_name}
            label_map = {int(k): v for k, v in json.loads(h5_file.attrs['label_map']).items()}
            # code_to_label: int_code -> project label (from events dict)
            code_to_label = {
                code: self.events[event_name]
                for code, event_name in label_map.items()
                if event_name in self.events
            }

            if not code_to_label:
                return

            # Channel selection index (global in v2 — same channels for all epochs)
            if self.channels_select:
                ch_names_arr = np.array(json.loads(h5_file.attrs['ch_names']))
                ch_idx = [int(np.where(ch_names_arr == ch)[0][0]) for ch in self.channels_select]
                self._ch_cache[None] = ch_idx

            self._available_epoch_meta_fields = set(h5_file['epoch_meta'].keys())
            event_codes = h5_file['epoch_meta/event_code'][:]
            source_groups = h5_file['epoch_meta/source_group'][:].astype(str)

            for flat_idx, (code, group_name) in enumerate(zip(event_codes, source_groups)):
                if int(code) in code_to_label:
                    self.index.append((group_name, flat_idx, code_to_label[int(code)]))
                    self._selected_flat_indices.append(flat_idx)

    def _get_h5_file(self):
        # Each process/worker lazily opens and owns its own read-only HDF5 handle.
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def close(self):
        """Close the HDF5 file"""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __getstate__(self):
        # Ensure opened HDF5 handle is never serialized to worker processes.
        self.close()
        state = self.__dict__.copy()
        state['_h5_file'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5_file = None

    def get_subjects(self):
        """Get subjects for all epochs without loading data"""
        return [get_subject_by_groupname(item[0], self.dataset_name) for item in self.index]

    def get_ch_names(self) -> list:
        """Return channel names from root attrs (v2 format)."""
        with h5py.File(self.h5_path, 'r') as f:
            return json.loads(f.attrs['ch_names'])

    def has_epoch_meta(self, field_name: str) -> bool:
        return field_name in self._available_epoch_meta_fields

    def get_epoch_meta(self, field_name: str, default=None):
        if field_name in self._epoch_meta_cache:
            return list(self._epoch_meta_cache[field_name])

        if not self.has_epoch_meta(field_name):
            return [default] * len(self.index)

        with h5py.File(self.h5_path, 'r') as f:
            dataset = f[f'epoch_meta/{field_name}']
            values = dataset[self._selected_flat_indices]
            if h5py.check_string_dtype(dataset.dtype) is not None:
                values = values.astype(str)
            self._epoch_meta_cache[field_name] = list(values)
        return list(self._epoch_meta_cache[field_name])

    def _get_epoch_meta_value(self, field_name: str, selected_idx: int):
        values = self.get_epoch_meta(field_name)
        return values[selected_idx]

    def __del__(self):
        # Module teardown can clear h5py internals before Dataset finalizers run.
        # Explicit ``close()`` calls should still surface real I/O errors, while a
        # best-effort destructor must never emit an ignored shutdown traceback.
        try:
            self.close()
        except Exception:
            pass


class EpochDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper to convert EpochDataset output to (data, label) format

    This maintains compatibility with code expecting TensorDataset format.
    """
    def __init__(self, subset: Subset):
        self.subset = subset
        self.dataset = subset.dataset
        self.indices = subset.indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data, label, ch_info, subject = self.subset[idx]
        # Convert label to tensor for compatibility
        label = torch.tensor(label, dtype=torch.long)
        return data, label

    def close(self):
        """Close the underlying HDF5 file"""
        if hasattr(self.dataset, 'close'):
            self.dataset.close()


class SubjectSampler(Sampler):
    def __init__(self, dataset, subject_ids):
        self.indices = [i for i in range(len(dataset)) if dataset[i][3] in subject_ids]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ClusterBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.cluster_to_indices = {}
        for i in range(len(dataset)):
            _, _, cluster_id = dataset[i]
            self.cluster_to_indices.setdefault(cluster_id, []).append(i)

        self.cluster_batches = []
        for indices in self.cluster_to_indices.values():
            if shuffle:
                np.random.shuffle(indices)

            batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
            self.cluster_batches.extend(batches)

        if shuffle:
            np.random.shuffle(self.cluster_batches)

    def __iter__(self):
        for batch in self.cluster_batches:
            yield batch

    def __len__(self):
        return len(self.cluster_batches)
