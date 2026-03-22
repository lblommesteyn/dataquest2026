"""Serialization helpers."""
import json
import os
from pathlib import Path

import joblib
import numpy as np


def save_pickle(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path: str):
    return joblib.load(path)


def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def save_arrays(path: str, **arrays):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_arrays(path: str):
    return dict(np.load(path, allow_pickle=True))
