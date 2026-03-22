import yaml
import os
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent
_PROJECT_ROOT = _CONFIG_DIR.parent


def load_config():
    base = yaml.safe_load((_CONFIG_DIR / "base_config.yaml").read_text())
    paths = yaml.safe_load((_CONFIG_DIR / "paths_config.yaml").read_text())
    # Resolve paths relative to project root
    for k, v in paths.items():
        if isinstance(v, str):
            paths[k] = str(_PROJECT_ROOT / v)
    base["paths"] = paths
    return base


CFG = load_config()
