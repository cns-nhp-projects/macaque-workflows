import json
from pathlib import Path


def check_params(params: str | Path | dict, required: set) -> dict:
    """Load and validate default or custom parameter file"""

    if not isinstance(params, dict):
        with open(params) as f:
            params = json.load(f)

    if not required <= set(params):
        raise KeyError(f"Parameters must have keys: {required}")

    return params
