"""Lightweight file-signature sniffing for upload MIME validation."""

from __future__ import annotations

from pathlib import Path

# Extension → acceptable sniffed MIME families.
_EXT_FAMILIES: dict[str, frozenset[str]] = {
    ".mp3": frozenset({"audio/mpeg", "audio/mp3"}),
    ".m4a": frozenset({"audio/mp4", "audio/m4a", "audio/x-m4a"}),
    ".wav": frozenset({"audio/wav", "audio/x-wav"}),
    ".ogg": frozenset({"audio/ogg", "video/ogg"}),
    ".oga": frozenset({"audio/ogg"}),
    ".webm": frozenset({"audio/webm", "video/webm"}),
    ".mp4": frozenset({"video/mp4", "audio/mp4"}),
    ".mov": frozenset({"video/quicktime", "video/mp4"}),
    ".mkv": frozenset({"video/x-matroska", "video/webm"}),
}

_CLAIM_ALIASES: dict[str, str] = {
    "audio/mp3": "audio/mpeg",
    "audio/x-wav": "audio/wav",
    "audio/x-m4a": "audio/mp4",
    "audio/m4a": "audio/mp4",
}


def sniff_mime(data: bytes) -> str | None:
    """Return a best-effort MIME from magic bytes, or None if unknown."""
    if not data:
        return None
    if data.startswith(b"ID3"):
        return "audio/mpeg"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "audio/mpeg"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "audio/wav"
    if data.startswith(b"OggS"):
        return "audio/ogg"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in {b"M4A ", b"M4B ", b"mp4a"}:
            return "audio/mp4"
        if brand in {b"qt  ", b"M4V "}:
            return "video/quicktime"
        return "video/mp4"
    if data.startswith(b"\x1aE\xdf\xa3"):
        # EBML: WebM or Matroska — accept either at the family layer.
        if b"webm" in data[:64].lower():
            return "video/webm"
        return "video/x-matroska"
    return None


def normalize_mime(value: str) -> str:
    lowered = value.strip().lower().split(";", 1)[0].strip()
    return _CLAIM_ALIASES.get(lowered, lowered)


def mime_compatible(
    *,
    filename: str,
    claimed_mime: str,
    head: bytes,
) -> bool:
    """True when sniffed signature matches extension / claimed MIME family."""
    sniffed = sniff_mime(head)
    if sniffed is None:
        return False
    sniffed_n = normalize_mime(sniffed)
    claimed_n = normalize_mime(claimed_mime) if claimed_mime else ""
    ext = Path(filename).suffix.lower()
    family = _EXT_FAMILIES.get(ext)
    if family is not None and sniffed_n not in family:
        return False
    if claimed_n and claimed_n != "application/octet-stream":
        claimed_family = family or frozenset({claimed_n})
        # Claimed type must sit in the same family as the extension/sniff.
        if claimed_n not in claimed_family and claimed_n != sniffed_n:
            # Allow alias overlap: audio/mpeg vs audio/mp3 already normalized.
            if family is not None and claimed_n not in family:
                return False
    return True
