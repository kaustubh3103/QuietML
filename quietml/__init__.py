"""
QuietML — A lightweight Python tool to silence and simplify noisy ML logs 
across frameworks like LightGBM, XGBoost, CatBoost, and TensorFlow.

Features:
- Unified log control with three modes: Silent, Smart, and Debug
- Temporary mode switching via a context manager
- Automatic detection and verbosity adjustment for supported frameworks
- Plug-and-play integration into any ML workflow

Example:
    from quietml import set_mode, silence, apply_to_model, configure

    set_mode("smart")  # Set global Smart mode
    with silence("silent"):  # Temporarily mute logs
        ...
"""

from .core import (
    set_mode,
    get_mode,
    silence,
    apply_to_model,
    configure,
)

__all__ = [
    "set_mode",
    "get_mode",
    "silence",
    "apply_to_model",
    "configure",
]
