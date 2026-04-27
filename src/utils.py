import importlib.util
import json
import random
from pathlib import Path

import numpy as np


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("project_config", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "CONFIG"):
            raise AttributeError(f"{config_path} must define CONFIG")
        return module.CONFIG

    if config_path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "PyYAML is required for YAML configs. Install requirements or use config.py."
            ) from error
        with config_path.open("r", encoding="utf-8") as file_obj:
            return yaml.safe_load(file_obj)

    raise ValueError(f"Unsupported config format: {config_path.suffix}")


def ensure_directories(paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def save_json(payload, output_path):
    def _default_serializer(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True, default=_default_serializer)
