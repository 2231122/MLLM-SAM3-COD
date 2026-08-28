"""Stage 1: Fine-grained Category Query (FCQ) with Qwen2.5-VL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from pipeline_utils import annotation_path, image_files, write_json


def parse_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "", 1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Qwen response did not contain a JSON object")
    return json.loads(text[start : end + 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True, help="Local Qwen2.5-VL checkpoint or Hugging Face model ID")
    parser.add_argument("--prompt", default="prompts/fine_grained_category_query.txt")
    parser.add_argument("--system-prompt", default="prompts/system.txt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model.to(device)
    processor = AutoProcessor.from_pretrained(args.model_path)
    task_prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8").strip()

    for image_path in tqdm(image_files(args.image_dir), desc="FCQ"):
        output_path = annotation_path(args.output_dir, image_path)
        if output_path.exists() and not args.overwrite:
            continue
        width, height = Image.open(image_path).size
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": task_prompt}]},
        ]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        response = processor.batch_decode(generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0]
        try:
            annotation = parse_json(response)
        except (ValueError, json.JSONDecodeError) as exc:
            annotation = {"error": str(exc), "raw_response": response}
        annotation.update({"image": image_path.name, "image_size": [width, height]})
        write_json(output_path, annotation)


if __name__ == "__main__":
    main()
