"""Utility helpers for surveillance components."""

import logging


LOGGER = logging.getLogger("surveillance")


def resolve_device() -> str:
    """Return the preferred torch device if available."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        LOGGER.warning("Torch not installed; defaulting to CPU.")
        return "cpu"
