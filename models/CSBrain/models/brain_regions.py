"""
Brain Region Computation Utility for CSBrain

Provides functions to map EEG channel names to brain regions and compute
sorted indices for brain-aware EEG processing.

Region mapping follows the CSBrain original repository conventions.
Note: The original repo uses per-dataset hand-coded mappings which are
inconsistent for some channels (e.g., FC is FRONTAL in PhysioNet but
CENTRAL in SEED-V; PO is OCCIPITAL in PhysioNet but PARIETAL in SEED-V).
This module uses a unified generic mapping as default.
"""
from typing import Dict, List, Tuple


# Brain region encoding constants
FRONTAL, PARIETAL, TEMPORAL, OCCIPITAL, CENTRAL = 0, 1, 2, 3, 4


def _normalize_ch_name(ch: str) -> str:
    """
    Normalize channel name by removing common prefixes.

    Args:
        ch: Channel name string

    Returns:
        Normalized channel name
    """
    ch = ch.strip()

    for p in ("EEG:", "eeg:", "EEG ", "eeg ", "EEG-", "eeg-"):
        if ch.startswith(p):
            ch = ch[len(p):].strip()
            break
    return ch


def _region_id(label: str) -> int:
    """
    Map channel label to brain region ID.

    Args:
        label: Channel label (e.g., 'Fp1', 'C3', 'Oz')

    Returns:
        Region ID: FRONTAL(0), PARIETAL(1), TEMPORAL(2), OCCIPITAL(3), CENTRAL(4)
    """
    u = label.upper()

    # Explicit special channels
    if u in {"CB1", "CB2"}:
        return OCCIPITAL

    # Occipital: Iz, O*
    if u == "IZ" or u.startswith("O"):
        return OCCIPITAL

    # Parieto-Occipital: PO* -> OCCIPITAL (follows PhysioNet convention)
    if u.startswith("PO"):
        return OCCIPITAL

    # Parietal: TPP*, P*
    if u.startswith("TPP"):
        return PARIETAL
    if u.startswith("P"):
        return PARIETAL

    # Central: CCP*, Cz, C* (with digit), CP*, FC*
    # FC (Fronto-Central) is placed in CENTRAL following PhysioNet convention
    if u.startswith("CCP"):
        return CENTRAL
    if u == "CZ" or (u.startswith("C") and len(u) > 1 and u[1:].replace("Z", "").isdigit()) or u.startswith("CP"):
        return CENTRAL
    if u.startswith("FC"):
        return CENTRAL

    # Temporal: FTT*, TTP*, FT*, TP*, T*
    if u.startswith(("FTT", "TTP")):
        return TEMPORAL
    if u.startswith(("FT", "TP", "T")):
        return TEMPORAL

    # Frontal: Fp*, AF*, F*
    if u.startswith(("FP", "AF", "F")):
        return FRONTAL

    # Mastoid/auricular references: M1, M2, A1, A2
    if u in {"M1", "M2", "A1", "A2"}:
        return CENTRAL

    # Legacy temporal labels: T3, T4, T5, T6, T1, T2
    if u in {"T3", "T4", "T5", "T6", "T1", "T2"}:
        return TEMPORAL

    # Default to CENTRAL for unknown channels
    return CENTRAL


def compute_brain_regions(ch_names: List[str]) -> Tuple[Tuple[int, ...], List[int]]:
    """
    Compute brain region tuple and sorted indices for channel ordering.

    This function maps each channel to its brain region and sorts channels
    by region (Frontal -> Central -> Temporal -> Parietal -> Occipital).

    Args:
        ch_names: List of EEG channel names

    Returns:
        Tuple of:
            - brain_regions: Tuple of region IDs for each channel (in sorted order)
            - sorted_indices: List of indices to reorder channels by region
    """
    cleaned = [_normalize_ch_name(ch) for ch in ch_names]
    brain_regions_raw = tuple(_region_id(lab) for lab in cleaned)

    # Build topology dictionary using normalized names
    topology: Dict[int, List[str]] = {
        FRONTAL: [],
        PARIETAL: [],
        TEMPORAL: [],
        OCCIPITAL: [],
        CENTRAL: []
    }
    for lab in cleaned:
        topology[_region_id(lab)].append(lab)

    # Group channels by region, using normalized names for consistent sorting
    region_groups: Dict[int, List[Tuple[int, str]]] = {}
    for i, region in enumerate(brain_regions_raw):
        if region not in region_groups:
            region_groups[region] = []
        region_groups[region].append((i, cleaned[i]))

    # Sort by topological relationship using normalized names
    sorted_indices = []
    for region in sorted(region_groups.keys()):
        region_electrodes = region_groups[region]
        topo_order = topology.get(region, [])
        rank = {name: idx for idx, name in enumerate(topo_order)}

        sorted_electrodes = sorted(
            region_electrodes,
            key=lambda x: (rank.get(x[1], float("inf")), x[0])
        )

        sorted_indices.extend([e[0] for e in sorted_electrodes])

    # Return brain_regions in sorted order (as expected by generate_area_config)
    brain_regions_sorted = tuple(brain_regions_raw[i] for i in sorted_indices)
    return brain_regions_sorted, sorted_indices


def generate_area_config(brain_regions: Tuple[int, ...]) -> Dict:
    """
    Generate area configuration for brain-aware transformer.

    Args:
        brain_regions: Tuple of region IDs for each channel (must be in sorted order)

    Returns:
        Dictionary mapping region names to channel slice information
    """
    from collections import defaultdict

    region_to_channels = defaultdict(list)
    for channel_idx, region in enumerate(brain_regions):
        region_to_channels[region].append(channel_idx)

    area_config = {}
    for region, channels in region_to_channels.items():
        area_config[f'region_{region}'] = {
            'channels': len(channels),
            'slice': slice(channels[0], channels[-1] + 1)
        }
    return area_config
