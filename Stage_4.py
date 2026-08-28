"""Stage 4: Semantic-Geometric Reasoning Injection (SGRI) for hard samples.

Without --model-path this script only prepares grouped geometric clues.  Supplying
--model-path completes the Qwen re-reasoning, SAM3 re-segmentation, and final
mask selection loop.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from pipeline_utils import category_chain_terms, read_json, write_json


def iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def groups(masks: list[np.ndarray], threshold: float) -> list[list[int]]:
    result: list[list[int]] = []
    for index, mask in enumerate(masks):
        for group in result:
            merged = np.logical_or.reduce([masks[item] for item in group])
            if iou(mask, merged) >= threshold:
                group.append(index)
                break
        else:
            result.append([index])
    return result


def parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response did not contain a JSON object")
    return json.loads(text[start : end + 1])


def make_contact_sheet(image: np.ndarray, masks: list[np.ndarray], path: Path) -> None:
    """Draw every candidate contour with its one-based ID on one reasoning image."""
    canvas = image.copy()
    for index, mask in enumerate(masks, start=1):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = tuple(int(value) for value in np.random.default_rng(index).integers(80, 256, 3))
        cv2.drawContours(canvas, contours, -1, color, 2)
        ys, xs = np.where(mask)
        if len(xs):
            cv2.putText(canvas, str(index), (int(xs.min()), max(16, int(ys.min()))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    Image.fromarray(canvas).save(path)


def make_logit_clues(image: np.ndarray, logits: list[np.ndarray], fallback_masks: list[np.ndarray], path: Path) -> None:
    """Convert SAM3 probability maps into multi-level geometric contour clues."""
    canvas = image.copy()
    for index, (score_map, fallback) in enumerate(zip(logits, fallback_masks), start=1):
        if score_map is None:
            score_map = fallback.astype(np.float32)
        for threshold, color in ((0.25, (255, 180, 0)), (0.50, (0, 255, 0)), (0.75, (0, 120, 255))):
            contours, _ = cv2.findContours((score_map >= threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 1)
        ys, xs = np.where(score_map >= 0.5)
        if not len(xs):
            ys, xs = np.where(fallback)
        if len(xs):
            cv2.putText(canvas, str(index), (int(xs.min()), max(16, int(ys.min()))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    Image.fromarray(canvas).save(path)


def qwen_query(model, processor, process_vision_info, image_paths: list[Path], prompt: str) -> dict:
    content = [{"type": "image", "image": str(path)} for path in image_paths]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=512)
    response = processor.batch_decode(output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0]
    return parse_json(response)


def save_sam3_mask(output: dict, path: Path) -> np.ndarray | None:
    logits = output.get("masks_logits")
    masks = output.get("masks")
    if logits is None and (masks is None or masks.numel() == 0):
        return None
    if logits is not None and logits.numel() > 0:
        binary = (logits.squeeze(1).amax(dim=0) > 0.5).detach().cpu().numpy()
    else:
        binary = masks.squeeze(1).any(dim=0).detach().cpu().numpy().astype(bool)
    Image.fromarray((binary * 255).astype(np.uint8)).save(path)
    return binary


def make_mask_grid(masks: list[np.ndarray], path: Path) -> list[int]:
    """Compose a 2×2 grid and resize it back to the original image resolution."""
    ids = list(range(1, min(4, len(masks)) + 1))
    height, width = masks[0].shape
    grid = Image.new("RGB", (width, height), "black")
    for slot, candidate_id in enumerate(ids):
        tile = Image.fromarray((masks[candidate_id - 1] * 255).astype(np.uint8)).convert("RGB")
        tile = tile.resize((width // 2, height // 2), Image.Resampling.NEAREST)
        ImageDraw.Draw(tile).text((8, 8), str(candidate_id), fill="red", stroke_width=1, stroke_fill="white")
        grid.paste(tile, ((slot % 2) * (width // 2), (slot // 2) * (height // 2)))
    grid.save(path)
    return ids


def write_final_mask(mask: np.ndarray, path: Path, final_root: Path | None, image_stem: str) -> None:
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)
    if final_root:
        shutil.copy2(path, final_root / f"{image_stem}.png")


def segment_categories(processor, image: Image.Image, categories: list[str], result_dir: Path, prefix: str) -> tuple[list[np.ndarray], list[str]]:
    """Run every foreground category through the SAM3 logits branch and binarize it."""
    state = processor.set_image(image)
    masks, prompts = [], []
    for term in category_chain_terms({"camouflaged_target_category": categories}):
        for suffix in ("", " Camouflaged"):
            prompt = f"{term}{suffix}"
            mask = save_sam3_mask(processor.set_text_prompt(state=state.copy(), prompt=prompt), result_dir / f"{prefix}_{len(masks) + 1}.png")
            if mask is not None:
                masks.append(mask)
                prompts.append(prompt)
    return masks, prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--sam3-dir", required=True)
    parser.add_argument("--sgdc-dir", required=True)
    parser.add_argument("--annotation-dir", help="Stage 1 FCQ annotations; required for the complete SGRI loop")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--final-mask-dir", help="Optional unified directory for reliable and hard-sample final masks")
    parser.add_argument("--group-iou", type=float, default=0.6)
    parser.add_argument("--model-path", help="Qwen2.5-VL checkpoint; enables the complete SGRI loop")
    parser.add_argument("--checkpoint", default=None, help="Optional local SAM3 checkpoint")
    parser.add_argument("--strict-empty", action="store_true")
    args = parser.parse_args()
    if args.checkpoint and not args.model_path:
        parser.error("--checkpoint requires --model-path; omit both for preparation only")
    if args.model_path and not args.annotation_dir:
        parser.error("--annotation-dir is required with --model-path")

    hard = set(read_json(Path(args.sgdc_dir) / "hard_samples.json", []))
    output_root = Path(args.output_dir)
    final_mask_root = Path(args.final_mask_dir) if args.final_mask_dir else None
    if final_mask_root:
        final_mask_root.mkdir(parents=True, exist_ok=True)
        for confirmed in Path(args.sgdc_dir).glob("*/fused_mask.png"):
            shutil.copy2(confirmed, final_mask_root / f"{confirmed.parent.name}.png")
    full_loop = args.model_path is not None
    if full_loop:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto")
        qwen_processor = AutoProcessor.from_pretrained(args.model_path)
        sam3_root = Path(__file__).parent / "sam3-main"
        sys.path.insert(0, str(sam3_root))
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam3 = Sam3Processor(build_sam3_image_model(checkpoint_path=args.checkpoint, device=device), device=device,
                            keep_best_when_empty=not args.strict_empty)

    library_path = Path(args.sgdc_dir) / "category_library.txt"
    library = library_path.read_text(encoding="utf-8") if library_path.exists() else ""
    category_prompt = (
        "This image contains numbered contour hints for possible camouflaged animals. "
        "Use them only as weak spatial evidence. Infer the camouflaged animal categories. "
        f"Previously confirmed category examples:\n{library[:6000]}\n"
        "Return only JSON: {\"camouflaged_target_category_inferred\": [\"coarse\", \"fine\"]}."
    )

    for image_name in tqdm(sorted(hard), desc="SGRI"):
        image_path = Path(args.image_dir) / image_name
        manifest = read_json(Path(args.sam3_dir) / image_path.stem / "manifest.json", {})
        candidates = manifest.get("foreground", [])
        fallback_masks = [np.asarray(Image.open(item["mask"]).convert("L")) > 0 for item in candidates]
        if not image_path.exists():
            continue
        result_dir = output_root / image_path.stem
        result_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        source = np.asarray(image)
        initial_prompts = []
        if full_loop:
            annotation = read_json(Path(args.annotation_dir) / f"{image_path.stem}.json", {})
            initial_masks, initial_prompts = segment_categories(
                sam3, image, annotation.get("camouflaged_target_category", []), result_dir, "initial_logits"
            )
            initial_masks = initial_masks or fallback_masks
        else:
            initial_masks = fallback_masks
        if not initial_masks:
            final_path = result_dir / "final_mask.png"
            write_final_mask(np.zeros(source.shape[:2], dtype=bool), final_path, final_mask_root, image_path.stem)
            write_json(result_dir / "reasoning_input.json", {"image": str(image_path), "mode": "empty_sam3_candidates", "final_mask": str(final_path)})
            continue
        combined_prior = np.logical_or.reduce(initial_masks)
        initial_clue = result_dir / "combined_geometric_prior.png"
        make_contact_sheet(source, [combined_prior], initial_clue)
        Image.fromarray((combined_prior * 255).astype(np.uint8)).save(result_dir / "combined_geometric_prior_mask.png")

        record = {"image": str(image_path), "initial_clues": str(initial_clue), "initial_prompts": initial_prompts, "mode": "prepared"}
        if not full_loop:
            final_path = result_dir / "final_mask.png"
            write_final_mask(combined_prior, final_path, final_mask_root, image_path.stem)
            record.update({"mode": "prepared_fallback", "final_mask": str(final_path)})
            write_json(result_dir / "reasoning_input.json", record)
            continue

        try:
            reasoning = qwen_query(qwen, qwen_processor, process_vision_info, [initial_clue], category_prompt)
        except (ValueError, json.JSONDecodeError) as exc:
            final_path = result_dir / "final_mask.png"
            write_final_mask(combined_prior, final_path, final_mask_root, image_path.stem)
            record.update({"mode": "reasoning_fallback", "error": str(exc), "final_mask": str(final_path)})
            write_json(result_dir / "reasoning_input.json", record)
            continue
        inferred = reasoning.get("camouflaged_target_category_inferred", [])
        if isinstance(inferred, str):
            inferred = [inferred]
        refined_masks, refined_prompts = segment_categories(sam3, image, inferred, result_dir, "refined_logits")
        candidate_masks = refined_masks or initial_masks
        final_groups = groups(candidate_masks, args.group_iou)
        final_masks = [np.logical_or.reduce([candidate_masks[index] for index in group]) for group in final_groups]
        final_clue = result_dir / "refined_clues.png"
        make_contact_sheet(source, final_masks, final_clue)
        mask_grid = result_dir / "mask_grid.png"
        grid_ids = make_mask_grid(final_masks, mask_grid)
        selection_prompt = (
            "The first image contains contour hints for a camouflaged animal. The second image is a 2x2 grid of "
            "candidate segmentation masks indexed as [1, 2; 3, 4]. Select the mask that best represents the animal. "
            "Return only JSON: {\"best_mask_id\": 1}."
        )
        try:
            selection = qwen_query(qwen, qwen_processor, process_vision_info, [initial_clue, mask_grid], selection_prompt)
        except (ValueError, json.JSONDecodeError):
            selection = {}
        selected = selection.get("best_mask_id")
        if not isinstance(selected, int) or selected not in grid_ids:
            selected = grid_ids[0]
        final_mask = final_masks[selected - 1]
        final_path = result_dir / "final_mask.png"
        write_final_mask(final_mask, final_path, final_mask_root, image_path.stem)
        record.update({"mode": "complete", "reasoning": reasoning, "refined_prompts": refined_prompts,
                       "refined_clues": str(final_clue), "mask_grid": str(mask_grid), "best_mask_id": selected,
                       "final_mask": str(final_path)})
        write_json(result_dir / "reasoning_input.json", record)


if __name__ == "__main__":
    main()
