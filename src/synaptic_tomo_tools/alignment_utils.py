"""Validation for alignment subdirectory names (no default alignment in this codebase)."""

from __future__ import annotations

from typing import Any


def require_alignment_dir(alignment_dir: Any, *, context: str = "") -> str:
    """
    Return a non-empty alignment directory name, or raise.

    Callers must obtain ``alignment_dir`` from the tomogram CSV (``alignment_dir`` column)
    or from an explicit API/CLI argument—never from a hardcoded default.
    """
    if alignment_dir is None:
        msg = "alignment_dir is required and cannot be None."
        if context:
            msg = f"{msg} ({context})"
        raise ValueError(msg)
    s = str(alignment_dir).strip()
    if not s or s.lower() in ("nan", "none"):
        msg = (
            "alignment_dir must be a non-empty string. "
            "Add an 'alignment_dir' column to your tomogram CSV for each row, "
            "or pass --alignment-dir explicitly for single-tomogram tools."
        )
        if context:
            msg = f"{msg} ({context})"
        raise ValueError(msg)
    return s
