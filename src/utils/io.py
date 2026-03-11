"""IO helpers."""
from pathlib import Path
import json


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(data, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
