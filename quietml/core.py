# quietml/core.py

"""
QuietML Core — the heart of QuietML.

A lightweight, framework-agnostic module that manages verbosity levels
for ML libraries like LightGBM, XGBoost, CatBoost, and TensorFlow.

Modes:
    💤 silent — hides all logs except critical errors
    🤖 smart  — keeps essential information and warnings
    🧩 debug  — shows everything (for developers)

Features:
- Global mode switching (`set_mode`)
- Temporary silence mode (`silence`)
- Per-model verbosity adjustment (`apply_to_model`)
- Combined helper for setup (`configure`)
"""

import os
from contextlib import contextmanager

# === Mode map for supported ML frameworks ===
MODE_MAP = {
    "silent": {
        "lightgbm": {"verbosity": -1},
        "xgboost": {"verbosity": 0},
        "catboost": {"logging_level": "Silent"},
        "tensorflow": {"TF_CPP_MIN_LOG_LEVEL": "3"},
    },
    "smart": {
        "lightgbm": {"verbosity": 1},
        "xgboost": {"verbosity": 1},
        "catboost": {"logging_level": "Info"},
        "tensorflow": {"TF_CPP_MIN_LOG_LEVEL": "2"},
    },
    "debug": {
        "lightgbm": {"verbosity": 2},
        "xgboost": {"verbosity": 3},
        "catboost": {"logging_level": "Verbose"},
        "tensorflow": {"TF_CPP_MIN_LOG_LEVEL": "0"},
    },
}

_current_mode = "smart"  # Default mode


# === Core Functions ===

def set_mode(mode: str):
    """
    Set the global QuietML mode.

    Args:
        mode (str): One of {"silent", "smart", "debug"}.

    Example:
        >>> set_mode("silent")
        [QuietML] Global mode set to: SILENT
    """
    global _current_mode
    mode = mode.lower()
    if mode not in MODE_MAP:
        raise ValueError(f"Invalid mode: {mode}. Choose from {list(MODE_MAP.keys())}")
    _current_mode = mode
    print(f"[QuietML] Global mode set to: {mode.upper()}")
    _apply_env(mode)


def get_mode():
    """Return the current QuietML mode."""
    return _current_mode


def _apply_env(mode: str):
    """Apply environment variables (e.g., TensorFlow log levels)."""
    config = MODE_MAP[mode]
    tf_level = config["tensorflow"]["TF_CPP_MIN_LOG_LEVEL"]
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = tf_level


def apply_to_model(model):
    """
    Automatically adjust an ML model's verbosity based on the current mode.

    Works with LightGBM, XGBoost, and CatBoost.

    Example:
        >>> model = apply_to_model(lgb.LGBMClassifier())
    """
    lib_name = model.__class__.__module__.split(".")[0].lower()
    if lib_name in MODE_MAP[_current_mode]:
        settings = MODE_MAP[_current_mode][lib_name]
        if "verbosity" in settings and hasattr(model, "set_params"):
            model.set_params(verbosity=settings["verbosity"])
        elif "logging_level" in settings and hasattr(model, "set_params"):
            model.set_params(logging_level=settings["logging_level"])
    return model


def configure(model, mode="smart"):
    """
    Set the global mode and apply it to a model in one call.

    Example:
        >>> model = configure(lgb.LGBMClassifier(), mode="smart")
    """
    set_mode(mode)
    return apply_to_model(model)


@contextmanager
def silence(mode="smart"):
    """
    Temporarily switch QuietML mode within a context block.

    Example:
        >>> with silence("silent"):
        ...     model.fit(X_train, y_train)
    """
    global _current_mode
    old_mode = _current_mode
    try:
        set_mode(mode)
        yield
    finally:
        set_mode(old_mode)
