"""Stage 2: obtain foreground and background SAM3 masks from FCQ annotations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pipeline_utils import annotation_path, category_chain_terms, image_files, read_json, write_json


def save_prediction(output: dict, path_base: Path, prompt: str) -> dict | None:
    masks = output.get("masks")
    logits = output.get("masks_logits")
    if masks is None or masks.numel() == 0:
        return None
    binary = masks.squeeze(1).any(dim=0).detach().cpu().numpy().astype(np.uint8) * 255
    mask_path = path_base.with_suffix(".png")
    Image.fromarray(binary).save(mask_path)
    record = {"prompt": prompt, "mask": str(mask_path)}
    if logits is not None:
        logits_path = path_base.with_suffix(".npy")
        np.save(logits_path, logits.squeeze(1).amax(dim=0).detach().float().cpu().numpy())
        record["logits"] = str(logits_path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional local SAM3 checkpoint")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--keep-best-when-empty", action="store_true", help="Retain the highest-scoring candidate when none pass SAM3 confidence filtering")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sam3_root = Path(__file__).parent / "sam3-main"
    sys.path.insert(0, str(sam3_root))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, device=device)
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.confidence_threshold,
        keep_best_when_empty=args.keep_best_when_empty,
    )
    output_root = Path(args.output_dir)

    for image_path in tqdm(image_files(args.image_dir), desc="SAM3"):
        annotation = read_json(annotation_path(args.annotation_dir, image_path), {})
        if annotation.get("error"):
            continue
        image_root = output_root / image_path.stem
        manifest_path = image_root / "manifest.json"
        if manifest_path.exists() and not args.overwrite:
            continue
        image_root.mkdir(parents=True, exist_ok=True)
        state = processor.set_image(Image.open(image_path).convert("RGB"))
        foreground, background = [], []
        for idx, category in enumerate(category_chain_terms(annotation)):
            for suffix in ("", " Camouflaged"):
                prompt = f"{category}{suffix}"
                prediction = processor.set_text_prompt(state=state.copy(), prompt=prompt)
                item = save_prediction(prediction, image_root / f"foreground_{idx:02d}_{suffix != ''}", prompt)
                if item:
                    foreground.append(item)
        bg_category = annotation.get("image_background_category", "")
        if isinstance(bg_category, str) and bg_category.strip():
            prediction = processor.set_text_prompt(state=state.copy(), prompt=bg_category.strip())
            item = save_prediction(prediction, image_root / "background", bg_category.strip())
            if item:
                background.append(item)
        write_json(manifest_path, {"image": str(image_path), "foreground": foreground, "background": background})


if __name__ == "__main__":
    main()
