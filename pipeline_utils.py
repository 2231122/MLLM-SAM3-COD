"""Small shared helpers for the four-stage MLLM-SAM3-COD pipeline."""

from __future__ import annotations

import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def read_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def annotation_path(annotation_dir: str | Path, image_path: str | Path) -> Path:
    return Path(annotation_dir) / f"{Path(image_path).stem}.json"


def category_chain_terms(annotation: dict) -> list[str]:
    terms: list[str] = []
    for chain in annotation.get("camouflaged_target_category", []):
        if not isinstance(chain, str):
            continue
        for term in chain.replace("Camouflaged", "").split("->"):
            term = term.strip()
            if term and term.lower() not in {item.lower() for item in terms}:
                terms.append(term)
    return terms
