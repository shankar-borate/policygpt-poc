"""Environment variable helpers.

Provides typed env-var readers (_env_int, _env_float, _env_bool) used by
Config.from_env() to build a Config from the process environment.

Adding a new env-var override:
  1. Add a reader call in Config.from_env() in settings.py.
  2. Document the variable name in .env.example.
"""

import os


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return default if not stripped else int(stripped)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return default if not stripped else float(stripped)


def _env_bool(name: str, default: bool | None) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip().lower()
    if not stripped:
        return default
    if stripped in {"1", "true", "yes", "on"}:
        return True
    if stripped in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value such as true/false or 1/0.")
