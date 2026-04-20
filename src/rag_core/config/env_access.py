"""Shared environment accessor helpers for workers modules.

Centralizes string/bool/int/float parsing so call sites avoid ad-hoc
``os.getenv``/``os.environ`` handling.
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def parse_env_bool(raw: str | None) -> bool | None:
    """Parse a raw env string into bool.

    Returns:
        - True/False for recognized values
        - None for missing or unrecognized values
    """
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    return None


def get_env(name: str, default: str | None = None) -> str | None:
    """Return raw env value, falling back to *default* when unset."""
    value = os.getenv(name)
    if value is None:
        return default
    return value


def get_env_stripped(name: str, default: str = "") -> str:
    """Return stripped env string with *default* fallback."""
    value = get_env(name, default)
    if value is None:
        return default.strip()
    return value.strip()


def get_env_optional(name: str) -> str | None:
    """Return raw env value or None when unset."""
    return get_env(name)


def get_env_int(name: str, default: int) -> int:
    """Parse env var as int; return *default* on invalid/missing."""
    raw = get_env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        # Invalid integer env values should fall back to the caller default instead of breaking startup.
        return default


def get_env_float(name: str, default: float) -> float:
    """Parse env var as float; return *default* on invalid/missing."""
    raw = get_env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        # Invalid float env values should fall back to the caller default instead of breaking startup.
        return default


def get_env_bool(name: str, default: bool) -> bool:
    """Parse env var as bool; return *default* on invalid/missing."""
    parsed = parse_env_bool(get_env(name))
    if parsed is None:
        return default
    return parsed


def get_env_optional_bool(name: str) -> bool | None:
    """Parse env var as bool; return None when missing or invalid."""
    return parse_env_bool(get_env(name))
