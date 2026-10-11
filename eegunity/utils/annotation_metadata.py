"""Helpers for encoding structured metadata into MNE annotation labels."""

from __future__ import annotations

from urllib.parse import quote, unquote


_ANNOTATION_METADATA_SEPARATOR = "||"


def encode_annotation_description(base_label: str, metadata: dict[str, object] | None = None, **kwargs) -> str:
    """Encode a base label plus scalar metadata into one annotation description."""
    label = str(base_label).strip()
    if not label:
        raise ValueError("base_label must be a non-empty string.")

    combined = {}
    if metadata:
        combined.update(metadata)
    combined.update(kwargs)

    parts = [label]
    for key in sorted(combined):
        key_text = str(key).strip()
        if not key_text:
            continue
        value = combined[key]
        if value is None:
            continue
        parts.append(f"{key_text}={quote(str(value), safe='._-/:')}")
    return _ANNOTATION_METADATA_SEPARATOR.join(parts)


def parse_annotation_description(description: str) -> tuple[str, dict[str, str]]:
    """Split an encoded annotation description into base label and metadata."""
    text = str(description)
    if _ANNOTATION_METADATA_SEPARATOR not in text:
        return text, {}

    tokens = text.split(_ANNOTATION_METADATA_SEPARATOR)
    base_label = tokens[0]
    metadata: dict[str, str] = {}
    for token in tokens[1:]:
        if "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        key = key.strip()
        if not key:
            continue
        metadata[key] = unquote(raw_value)
    return base_label, metadata
