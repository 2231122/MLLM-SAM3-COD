"""Stage 3: Semantic-Geometric Dual Confirmation (SGDC)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from pipeline_utils import image_files, read_json, write_json


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0


def mask_box(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def box_iou(left: list[int] | None, right: list[int]) -> float:
    if left is None or len(right) != 4:
        return 0.0
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_l = (left[2] - left[0] + 1) * (left[3] - left[1] + 1)
    area_r = (right[2] - right[0] + 1) * (right[3] - right[1] + 1)
    return inter / (area_l + area_r - inter) if area_l + area_r > inter else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--sam3-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--semantic-iou", type=float, default=0.3)
    parser.add_argument("--geometric-iou", type=float, default=0.3)
    args = parser.parse_args()
    output_root = Path(args.output_dir)
    library: set[str] = set()
    hard_samples = []

    for image_path in tqdm(image_files(args.image_dir), desc="SGDC"):
        manifest = read_json(Path(args.sam3_dir) / image_path.stem / "manifest.json", {})
        annotation = read_json(Path(args.annotation_dir) / f"{image_path.stem}.json", {})
        boxes = annotation.get("location_of_camouflaged_target", {}).get("bboxes", [])
        bg_masks = [np.asarray(Image.open(item["mask"]).convert("L")) > 0 for item in manifest.get("background", [])]
        accepted = []
        for item in manifest.get("foreground", []):
            mask = np.asarray(Image.open(item["mask"]).convert("L")) > 0
            components = cv2.connectedComponents(mask.astype(np.uint8))[0] - 1
            overlap = max((mask_iou(mask, bg) for bg in bg_masks), default=0.0)
            predicted_box = mask_box(mask)
            height, width = mask.shape
            covers_image = predicted_box is not None and (predicted_box[2] - predicted_box[0]) / width > 0.95 and (predicted_box[3] - predicted_box[1]) / height > 0.95
            geometry = max((box_iou(predicted_box, box) for box in boxes if len(box) == 4), default=0.0)
            if overlap <= args.semantic_iou and geometry >= args.geometric_iou and components <= 50 and not covers_image:
                accepted.append(item)
        result_dir = output_root / image_path.stem
        result_dir.mkdir(parents=True, exist_ok=True)
        if accepted:
            merged = np.zeros_like(np.asarray(Image.open(accepted[0]["mask"]).convert("L")), dtype=bool)
            for item in accepted:
                merged |= np.asarray(Image.open(item["mask"]).convert("L")) > 0
                shutil.copy2(item["mask"], result_dir / Path(item["mask"]).name)
            Image.fromarray((merged * 255).astype(np.uint8)).save(result_dir / "fused_mask.png")
            for category in annotation.get("camouflaged_target_category", []):
                if isinstance(category, str):
                    library.add(category)
        else:
            hard_samples.append(image_path.name)
        write_json(result_dir / "confirmation.json", {"image": image_path.name, "accepted": accepted, "hard": not bool(accepted)})
    write_json(output_root / "hard_samples.json", hard_samples)
    (output_root / "category_library.txt").write_text("\n".join(sorted(library)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
