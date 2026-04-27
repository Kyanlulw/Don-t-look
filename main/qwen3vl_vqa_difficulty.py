import argparse
import inspect
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)


SYSTEM_PROMPT = (
    "You are an expert evaluator for Vietnamese scene-text VQA difficulty. "
    "You must return exactly one valid JSON object and nothing else."
)

DEFAULT_WEIGHTS_FILE = Path(__file__).with_name("difficulty_weights.json")

WEIGHTS = {
    "text_visibility": 0.25,
    "ocr_ambiguity": 0.20,
    "linguistic_complexity": 0.20,
    "text_density": 0.15,
    "reasoning_required": 0.12,
    "text_orientation": 0.08,
}


def build_scoring_prompt(question: str, answer: str) -> str:
    return f"""Evaluate OCR-VQA difficulty for this sample.
Analyze the image carefully before scoring.

Question: {question}
Ground-truth answer: {answer}

Score each dimension from 1 (easiest) to 5 (hardest):
- text_visibility
- text_orientation
- text_density
- linguistic_complexity
- reasoning_required
- ocr_ambiguity

Return ONLY valid JSON using full keys exactly (no abbreviations):
{{
  "scores": {{
    "text_visibility": <1-5>,
    "text_orientation": <1-5>,
    "text_density": <1-5>,
    "linguistic_complexity": <1-5>,
    "reasoning_required": <1-5>,
    "ocr_ambiguity": <1-5>
  }}
}}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score OCR-VQA difficulty with Qwen3-VL and write JSONL results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input file (.jsonl with flat rows or ViTextVQA-style .json).",
    )
    parser.add_argument(
        "--input-2",
        type=Path,
        default=None,
        help="Optional second dataset (.jsonl or ViTextVQA-style .json).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional root directory prepended to relative image paths.",
    )
    parser.add_argument(
        "--image-root-2",
        type=Path,
        default=None,
        help="Optional image root for --input-2. Falls back to --image-root when omitted.",
    )
    parser.add_argument("--image-key", default="image", help="Image path field name.")
    parser.add_argument("--question-key", default="question", help="Question field name.")
    parser.add_argument("--answer-key", default="answer", help="Answer field name.")
    parser.add_argument(
        "--id-key",
        default="id",
        help="Optional id field used for resume and output tracking.",
    )
    parser.add_argument(
        "--weights-file",
        type=Path,
        default=DEFAULT_WEIGHTS_FILE,
        help="JSON file containing the per-dimension weights for weighted_difficulty.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size. Use a positive integer; larger values may OOM.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=192,
        help="Token budget for JSON output.",
    )
    parser.add_argument(
        "--short-side",
        type=int,
        default=768,
        help="Only downscale images whose short side exceeds this value.",
    )
    parser.add_argument(
        "--long-side",
        type=int,
        default=1280,
        help="Only downscale images whose long side exceeds this value.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=768 * 1280,
        help="Safety cap after downscaling to bound vision cost.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="T4 should use float16.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "eager"],
        help="Use sdpa by default.",
    )
    parser.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable bitsandbytes 4-bit loading.",
    )
    parser.add_argument(
        "--double-quant",
        action="store_true",
        help="Enable nested quantization for extra VRAM headroom.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen reasoning mode if supported by processor.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["qwen", "vintern"],
        default="qwen",
        help="Prompt scaffolding style. Use 'vintern' for InternVL-style '<image>\\n...' single-turn text prompts.",
    )
    parser.add_argument(
        "--model-backend",
        choices=["auto", "qwen", "vintern"],
        default="auto",
        help="Model loading backend. 'auto' infers from prompt style and model id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset sampling/mixing and torch RNG.",
    )
    parser.add_argument(
        "--source-tag-1",
        default="dataset1",
        help="Source tag for the first dataset in mixed runs.",
    )
    parser.add_argument(
        "--source-tag-2",
        default="dataset2",
        help="Source tag for the second dataset in mixed runs.",
    )
    parser.add_argument(
        "--range-1",
        default=None,
        help="Slice for first dataset as start:end (0-based, end exclusive).",
    )
    parser.add_argument(
        "--range-2",
        default=None,
        help="Slice for second dataset as start:end (0-based, end exclusive).",
    )
    parser.add_argument(
        "--pick-1",
        type=int,
        default=None,
        help="Randomly pick N samples from the ranged first dataset.",
    )
    parser.add_argument(
        "--pick-2",
        type=int,
        default=None,
        help="Randomly pick N samples from the ranged second dataset.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for quick tests.")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Flush output every N samples.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print throughput every N samples.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    return args


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def build_quant_config(args: argparse.Namespace) -> Optional[BitsAndBytesConfig]:
    if args.disable_4bit:
        return None
    compute_dtype = resolve_dtype(args.dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.double_quant,
    )


def resolve_model_backend(args: argparse.Namespace) -> str:
    if args.model_backend != "auto":
        return args.model_backend
    if args.prompt_style == "vintern":
        return "vintern"

    model_name = args.model.lower()
    if "vintern" in model_name or "internvl" in model_name:
        return "vintern"
    return "qwen"


def load_model_and_processor(
    args: argparse.Namespace,
) -> Tuple[Any, Any]:
    backend = resolve_model_backend(args)
    quantization_config = build_quant_config(args)
    torch_dtype = resolve_dtype(args.dtype)

    if backend == "vintern":
        if quantization_config is not None:
            logging.warning("4-bit quantization is ignored for Vintern backend.")

        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,
        )
        model = model.eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
            use_fast=False,
        )
        return model, tokenizer

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=28 * 28,
        max_pixels=args.max_pixels,
    )
    model.eval()
    return model, processor


def load_done_ids(output_path: Path, id_key: str) -> set:
    done = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = row.get("_resume_id", row.get(id_key))
            if sample_id is None or row.get("error"):
                continue
            model_output = row.get("model_output")
            if not isinstance(model_output, str):
                continue
            try:
                parse_model_json(model_output)
            except Exception:
                continue
            done.add(str(sample_id))
    return done


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc


def iter_vitextvqa_json(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(
            "Expected top-level 'images' and 'annotations' lists for ViTextVQA JSON input."
        )

    images_by_id = {}
    for image in images:
        if isinstance(image, dict) and "id" in image:
            images_by_id[image.get("id")] = image

    for ann_index, ann in enumerate(annotations, start=1):
        if not isinstance(ann, dict):
            continue

        image_id = ann.get("image_id")
        image_entry = images_by_id.get(image_id)

        if image_entry is None and isinstance(image_id, int) and 0 <= image_id < len(images):
            maybe_image = images[image_id]
            if isinstance(maybe_image, dict):
                image_entry = maybe_image

        if image_entry is None:
            raise ValueError(
                f"Could not resolve image for annotation index {ann_index} with image_id={image_id}."
            )

        image_value = image_entry.get("filename") or image_entry.get("image")
        if not image_value:
            raise ValueError(
                f"Missing image filename for annotation index {ann_index} (image_id={image_id})."
            )

        answers = ann.get("answers")
        if isinstance(answers, list) and answers:
            answer_value = answers[0]
        else:
            answer_value = ann.get("answer", "")

        yield {
            "id": ann.get("id", ann_index),
            "image": str(image_value),
            "question": str(ann.get("question", "")),
            "answer": str(answer_value),
        }


def iter_input_samples(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        yield from iter_jsonl(path)
        return

    if path.suffix.lower() == ".json":
        yield from iter_vitextvqa_json(path)
        return

    raise ValueError(
        f"Unsupported input format for {path}. Use .jsonl or ViTextVQA-style .json"
    )


def parse_slice_range(spec: Optional[str], total: int) -> Tuple[int, int]:
    if spec is None:
        return 0, total
    if ":" not in spec:
        raise ValueError(f"Invalid range '{spec}'. Expected format start:end")

    start_raw, end_raw = spec.split(":", 1)
    start = int(start_raw) if start_raw else 0
    end = int(end_raw) if end_raw else total

    if start < 0 or end < 0:
        raise ValueError(f"Invalid range '{spec}'. Negative indexes are not supported.")

    start = min(start, total)
    end = min(end, total)
    if start > end:
        raise ValueError(f"Invalid range '{spec}'. Start must be <= end.")
    return start, end


def build_resume_id(
    sample: Dict[str, Any],
    id_key: str,
    source_tag: str,
    row_index: int,
    use_source_prefix: bool,
) -> str:
    raw_id = sample.get(id_key)
    base_id = str(raw_id) if raw_id is not None else f"idx-{row_index}"
    if use_source_prefix:
        return f"{source_tag}:{base_id}"
    return base_id


def prepare_dataset_samples(
    path: Path,
    image_root: Optional[Path],
    range_spec: Optional[str],
    pick_count: Optional[int],
    source_tag: str,
    use_source_prefix: bool,
    done_ids: set,
    id_key: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    all_rows = list(iter_input_samples(path))
    start, end = parse_slice_range(range_spec, len(all_rows))
    ranged_rows = all_rows[start:end]

    prepared: List[Dict[str, Any]] = []
    for offset, row in enumerate(ranged_rows):
        original_index = start + offset
        resume_id = build_resume_id(
            sample=row,
            id_key=id_key,
            source_tag=source_tag,
            row_index=original_index,
            use_source_prefix=use_source_prefix,
        )
        if resume_id in done_ids:
            continue

        sample = dict(row)
        sample["_source"] = source_tag
        sample["_resume_id"] = resume_id
        sample["__image_root"] = image_root
        prepared.append(sample)

    if pick_count is not None:
        if pick_count < 0:
            raise ValueError("pick count must be >= 0")
        if pick_count < len(prepared):
            prepared = rng.sample(prepared, pick_count)
        elif pick_count > len(prepared):
            logging.warning(
                "Requested pick=%d from %s but only %d samples are available after filters.",
                pick_count,
                source_tag,
                len(prepared),
            )

    return prepared


def resolve_image_path(raw_path: str, image_root: Optional[Path]) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if image_root is not None:
        return image_root / path
    return path


def maybe_resize_image(
    image: Image.Image,
    short_side_limit: int,
    long_side_limit: int,
    max_pixels: int,
) -> Image.Image:
    width, height = image.size
    short_side = min(width, height)
    long_side = max(width, height)

    scale = 1.0
    if short_side > short_side_limit:
        scale = min(scale, short_side_limit / short_side)
    if long_side > long_side_limit:
        scale = min(scale, long_side_limit / long_side)
    if width * height > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / float(width * height)))

    if scale >= 1.0:
        return image

    new_width = max(28, int(round(width * scale)))
    new_height = max(28, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def load_image(
    path: Path,
    short_side_limit: int,
    long_side_limit: int,
    max_pixels: int,
) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        return maybe_resize_image(image, short_side_limit, long_side_limit, max_pixels)


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio_w, ratio_h in target_ratios:
        target_aspect_ratio = ratio_w / ratio_h
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = (ratio_w, ratio_h)
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio_w * ratio_h:
                best_ratio = (ratio_w, ratio_h)
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    sorted_ratios = sorted(target_ratios, key=lambda ratio: ratio[0] * ratio[1])

    ratio_w, ratio_h = find_closest_aspect_ratio(
        aspect_ratio=aspect_ratio,
        target_ratios=sorted_ratios,
        width=orig_width,
        height=orig_height,
        image_size=image_size,
    )

    target_width = image_size * ratio_w
    target_height = image_size * ratio_h
    blocks = ratio_w * ratio_h
    resized_img = image.resize((target_width, target_height), Image.Resampling.BICUBIC)

    processed_images: List[Image.Image] = []
    columns = target_width // image_size
    for index in range(blocks):
        box = (
            (index % columns) * image_size,
            (index // columns) * image_size,
            ((index % columns) + 1) * image_size,
            ((index // columns) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))

    return processed_images


def pil_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    tensor = tensor.view(image.height, image.width, 3).permute(2, 0, 1).float().div_(255.0)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def load_vintern_pixel_values(
    path: Path,
    target_dtype: torch.dtype,
    device: torch.device,
    input_size: int = 448,
    max_num: int = 6,
) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        tiles = dynamic_preprocess(
            image=image,
            image_size=input_size,
            max_num=max_num,
            use_thumbnail=True,
        )

    tile_tensors = [pil_to_normalized_tensor(tile) for tile in tiles]
    pixel_values = torch.stack(tile_tensors)
    return pixel_values.to(dtype=target_dtype, device=device)


def build_qwen_messages(question: str, answer: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": build_scoring_prompt(question, answer)},
            ],
        },
    ]


def build_vintern_question(question: str, answer: str) -> str:
    # Vintern (InternVL-style chat API) expects a single question string prefixed by <image>.
    return f"<image>\n{build_scoring_prompt(question, answer)}"


def build_prompt_text(
    processor: Optional[Any],
    question: str,
    answer: str,
    prompt_style: str,
    thinking_kwargs: Dict[str, Any],
) -> str:
    if prompt_style == "vintern":
        return build_vintern_question(question, answer)

    if processor is None:
        raise ValueError("Qwen prompt style requires a processor.")

    messages = build_qwen_messages(question, answer)
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **thinking_kwargs,
    )


def chunked(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def infer_input_device(model: Any) -> torch.device:
    if hasattr(model, "device") and str(model.device) != "meta":
        return model.device
    return next(model.parameters()).device


def extract_json_candidate(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(stripped[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    raise ValueError("Could not find a balanced JSON object in model output.")


def infer_tier(weighted_difficulty: float) -> str:
    if weighted_difficulty < 2.0:
        return "easy"
    if weighted_difficulty < 3.0:
        return "medium"
    if weighted_difficulty < 4.0:
        return "hard"
    return "very_hard"


def load_weights(weights_file: Path) -> Dict[str, float]:
    with weights_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Weights file must contain a JSON object: {weights_file}")

    weights: Dict[str, float] = {}
    for key in WEIGHTS:
        if key not in payload:
            raise ValueError(f"Missing weight key '{key}' in {weights_file}")
        try:
            weights[key] = float(payload[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Weight '{key}' must be numeric in {weights_file}") from exc

    total_weight = sum(weights.values())
    if not math.isclose(total_weight, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"Weights in {weights_file} must sum to 1.0, got {total_weight:.6f}"
        )

    return weights


def compute_weighted_difficulty(scores: Dict[str, int]) -> float:
    missing = [key for key in WEIGHTS if key not in scores]
    if missing:
        raise ValueError(f"Missing score keys for weighting: {', '.join(missing)}")

    weighted_sum = sum(scores[key] * weight for key, weight in WEIGHTS.items())
    return round(weighted_sum, 2)


def parse_model_json(text: str) -> Dict[str, Any]:
    payload = json.loads(extract_json_candidate(text))

    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("Missing 'scores' object in model output.")

    score_keys = [
        "text_visibility",
        "text_orientation",
        "text_density",
        "linguistic_complexity",
        "reasoning_required",
        "ocr_ambiguity",
    ]

    normalized_scores: Dict[str, int] = {}
    for key in score_keys:
        if key not in scores:
            raise ValueError(f"Missing score key: {key}")
        try:
            value = int(round(float(scores[key])))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Score {key} must be numeric.") from exc
        if value < 1 or value > 5:
            raise ValueError(f"Score {key} out of range: {value}")
        normalized_scores[key] = value

    weighted_difficulty = compute_weighted_difficulty(normalized_scores)
    tier = infer_tier(weighted_difficulty)

    return {
        "scores": normalized_scores,
        "weighted_difficulty": weighted_difficulty,
        "difficulty_tier": tier,
    }


def build_output_record(
    sample: Dict[str, Any],
    score: Optional[Dict[str, Any]],
    raw_output: str,
    error: Optional[str],
) -> Dict[str, Any]:
    result = {key: value for key, value in sample.items() if key != "__image_root"}
    result["model_output"] = raw_output
    if score is not None:
        result["difficulty"] = score
        result["weighted_difficulty"] = score["weighted_difficulty"]
        result["difficulty_tier"] = score["difficulty_tier"]
        result["scores"] = score["scores"]
    if error is not None:
        result["error"] = error
    return result


def is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message


def run_batch_generation(
    model: Any,
    processor: Any,
    model_inputs: Dict[str, Any],
    max_new_tokens: int,
) -> List[str]:
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
        )

    prompt_lengths = model_inputs["input_ids"].shape[1]
    trimmed_ids = generated_ids[:, prompt_lengths:]
    return processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def run_vintern_batch_generation(
    model: Any,
    tokenizer: Any,
    pixel_values_batch: List[torch.Tensor],
    prompts: List[str],
    max_new_tokens: int,
) -> List[str]:
    outputs: List[str] = []
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 3,
        "repetition_penalty": 2.5,
    }

    with torch.inference_mode():
        for pixel_values, prompt in zip(pixel_values_batch, prompts):
            response = model.chat(
                tokenizer,
                pixel_values,
                prompt,
                generation_config,
                history=None,
                return_history=False,
            )
            if isinstance(response, tuple):
                outputs.append(str(response[0]))
            else:
                outputs.append(str(response))
    return outputs


def prepare_model_inputs(
    processor: Any,
    texts: List[str],
    images: List[Image.Image],
    input_device: torch.device,
) -> Dict[str, Any]:
    model_inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    model_inputs.pop("token_type_ids", None)
    return {
        key: value.to(input_device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }


def should_retry_parse_error(raw_output: str, error: Exception) -> bool:
    msg = str(error).lower()
    hints = [
        "no json object",
        "balanced json",
        "expecting",
        "missing",
        "invalid",
        "out of range",
        "must be numeric",
    ]
    if any(hint in msg for hint in hints):
        return True
    return raw_output.strip() == ""


def build_retry_text(base_prompt_text: str) -> str:
    return (
        base_prompt_text
        + "\n\nIMPORTANT: Your previous reply was invalid. "
        + "Return exactly one complete JSON object with the full key names. "
        + "Do not add any explanation, markdown, or extra text."
    )


def main() -> None:
    global WEIGHTS
    args = parse_args()
    setup_logging()
    WEIGHTS = load_weights(args.weights_file)

    if args.input_2 is None and (
        args.image_root_2 is not None or args.range_2 is not None or args.pick_2 is not None
    ):
        raise ValueError("--image-root-2/--range-2/--pick-2 require --input-2")

    rng = random.Random(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. A T4 is expected here.")

    torch.cuda.manual_seed_all(args.seed)

    torch.set_grad_enabled(False)
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    done_ids = load_done_ids(args.output, args.id_key)
    use_source_prefix = args.input_2 is not None

    first_dataset_samples = prepare_dataset_samples(
        path=args.input,
        image_root=args.image_root,
        range_spec=args.range_1,
        pick_count=args.pick_1,
        source_tag=args.source_tag_1,
        use_source_prefix=use_source_prefix,
        done_ids=done_ids,
        id_key=args.id_key,
        rng=rng,
    )

    second_dataset_samples: List[Dict[str, Any]] = []
    if args.input_2 is not None:
        second_dataset_samples = prepare_dataset_samples(
            path=args.input_2,
            image_root=args.image_root_2 if args.image_root_2 is not None else args.image_root,
            range_spec=args.range_2,
            pick_count=args.pick_2,
            source_tag=args.source_tag_2,
            use_source_prefix=True,
            done_ids=done_ids,
            id_key=args.id_key,
            rng=rng,
        )

    all_samples = first_dataset_samples + second_dataset_samples
    if args.input_2 is not None:
        rng.shuffle(all_samples)

    if args.limit is not None:
        all_samples = all_samples[: args.limit]

    if not all_samples:
        logging.info("No pending samples found.")
        return

    backend = resolve_model_backend(args)
    model, processor = load_model_and_processor(args)
    input_device = infer_input_device(model) if backend == "qwen" else torch.device("cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Pending samples: %d", len(all_samples))
    if args.input_2 is not None:
        logging.info(
            "Mix summary: %s=%d, %s=%d, seed=%d",
            args.source_tag_1,
            len(first_dataset_samples),
            args.source_tag_2,
            len(second_dataset_samples),
            args.seed,
        )
    else:
        logging.info("Sampling seed: %d", args.seed)
    logging.info("Model: %s", args.model)
    logging.info("Backend: %s", backend)
    logging.info("Weights file: %s", args.weights_file)
    logging.info(
        "Config: 4bit=%s batch_size=%d max_new_tokens=%d short_side=%d long_side=%d max_pixels=%d",
        not args.disable_4bit,
        args.batch_size,
        args.max_new_tokens,
        args.short_side,
        args.long_side,
        args.max_pixels,
    )

    thinking_kwargs: Dict[str, Any] = {}
    if backend == "qwen":
        try:
            sig = inspect.signature(processor.apply_chat_template)
            if "enable_thinking" in sig.parameters:
                thinking_kwargs["enable_thinking"] = args.enable_thinking
        except Exception:
            pass

    started_at = time.time()
    written = 0
    error_rows = 0

    with args.output.open("a", encoding="utf-8") as out_handle:
        with tqdm(total=len(all_samples), desc="Scoring", unit="sample") as progress:
            for batch_index, batch in enumerate(chunked(all_samples, args.batch_size), start=1):
                image_payloads: List[Any] = []
                texts = []
                for sample in batch:
                    sample_image_root = sample.get("__image_root")
                    if sample_image_root is None:
                        sample_image_root = args.image_root
                    image_path = resolve_image_path(sample[args.image_key], sample_image_root)
                    if backend == "qwen":
                        image_payloads.append(
                            load_image(
                                image_path,
                                short_side_limit=args.short_side,
                                long_side_limit=args.long_side,
                                max_pixels=args.max_pixels,
                            )
                        )
                    else:
                        image_payloads.append(
                            load_vintern_pixel_values(
                                path=image_path,
                                target_dtype=resolve_dtype(args.dtype),
                                device=input_device,
                            )
                        )
                    texts.append(
                        build_prompt_text(
                            processor=processor if backend == "qwen" else None,
                            question=str(sample[args.question_key]),
                            answer=str(sample[args.answer_key]),
                            prompt_style=args.prompt_style if backend == "qwen" else "vintern",
                            thinking_kwargs=thinking_kwargs,
                        )
                    )

                if backend == "qwen":
                    model_inputs = prepare_model_inputs(
                        processor=processor,
                        texts=texts,
                        images=image_payloads,
                        input_device=input_device,
                    )

                    try:
                        outputs = run_batch_generation(
                            model=model,
                            processor=processor,
                            model_inputs=model_inputs,
                            max_new_tokens=args.max_new_tokens,
                        )
                    except Exception as exc:
                        if len(batch) == 1 or not is_oom_error(exc):
                            raise

                        logging.warning(
                            "OOM on batch %d with size %d. Retrying samples one by one.",
                            batch_index,
                            len(batch),
                        )
                        torch.cuda.empty_cache()
                        outputs = []
                        for single_image, single_text in zip(image_payloads, texts):
                            single_inputs = prepare_model_inputs(
                                processor=processor,
                                texts=[single_text],
                                images=[single_image],
                                input_device=input_device,
                            )
                            try:
                                single_output = run_batch_generation(
                                    model=model,
                                    processor=processor,
                                    model_inputs=single_inputs,
                                    max_new_tokens=args.max_new_tokens,
                                )[0]
                            except Exception as single_exc:
                                if is_oom_error(single_exc):
                                    torch.cuda.empty_cache()
                                    single_output = json.dumps(
                                        {"error": "cuda_oom_during_generation"},
                                        ensure_ascii=False,
                                    )
                                else:
                                    raise
                            outputs.append(single_output)
                else:
                    try:
                        outputs = run_vintern_batch_generation(
                            model=model,
                            tokenizer=processor,
                            pixel_values_batch=image_payloads,
                            prompts=texts,
                            max_new_tokens=args.max_new_tokens,
                        )
                    except Exception as exc:
                        if len(batch) == 1 or not is_oom_error(exc):
                            raise

                        logging.warning(
                            "OOM on batch %d with size %d. Retrying samples one by one.",
                            batch_index,
                            len(batch),
                        )
                        torch.cuda.empty_cache()
                        outputs = []
                        for single_image, single_text in zip(image_payloads, texts):
                            try:
                                single_output = run_vintern_batch_generation(
                                    model=model,
                                    tokenizer=processor,
                                    pixel_values_batch=[single_image],
                                    prompts=[single_text],
                                    max_new_tokens=args.max_new_tokens,
                                )[0]
                            except Exception as single_exc:
                                if is_oom_error(single_exc):
                                    torch.cuda.empty_cache()
                                    single_output = json.dumps(
                                        {"error": "cuda_oom_during_generation"},
                                        ensure_ascii=False,
                                    )
                                else:
                                    raise
                            outputs.append(single_output)

                for sample, raw_output, image_payload, sample_text in zip(
                    batch,
                    outputs,
                    image_payloads,
                    texts,
                ):
                    error = None
                    parsed = None
                    try:
                        parsed = parse_model_json(raw_output)
                    except Exception as exc:
                        error = str(exc)
                        if should_retry_parse_error(raw_output, exc):
                            retry_tokens = max(args.max_new_tokens * 2, 192)
                            logging.warning(
                                "Retrying invalid JSON for sample id=%s with max_new_tokens=%d",
                                sample.get(args.id_key),
                                retry_tokens,
                            )
                            try:
                                retry_text = build_retry_text(sample_text)
                                if backend == "qwen":
                                    retry_inputs = prepare_model_inputs(
                                        processor=processor,
                                        texts=[retry_text],
                                        images=[image_payload],
                                        input_device=input_device,
                                    )
                                    retry_output = run_batch_generation(
                                        model=model,
                                        processor=processor,
                                        model_inputs=retry_inputs,
                                        max_new_tokens=retry_tokens,
                                    )[0]
                                else:
                                    retry_output = run_vintern_batch_generation(
                                        model=model,
                                        tokenizer=processor,
                                        pixel_values_batch=[image_payload],
                                        prompts=[retry_text],
                                        max_new_tokens=retry_tokens,
                                    )[0]
                                parsed = parse_model_json(retry_output)
                                raw_output = retry_output
                                error = None
                            except Exception as retry_exc:
                                error = f"{error} | retry_failed: {retry_exc}"

                    if error is not None:
                        error_rows += 1

                    record = build_output_record(sample, parsed, raw_output, error)
                    out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    progress.update(1)

                if written % args.flush_every == 0:
                    out_handle.flush()

                if written % args.log_every == 0:
                    elapsed = max(time.time() - started_at, 1e-6)
                    speed = written / elapsed
                    progress.set_postfix(errors=error_rows, samples_per_sec=f"{speed:.2f}")
                    logging.info(
                        "Processed %d/%d samples | %.2f samples/s",
                        written,
                        len(all_samples),
                        speed,
                    )

                for image_payload in image_payloads:
                    if isinstance(image_payload, Image.Image):
                        image_payload.close()

    elapsed = max(time.time() - started_at, 1e-6)
    logging.info(
        "Finished %d samples in %.2f minutes (%.2f samples/s)",
        written,
        elapsed / 60.0,
        written / elapsed,
    )


if __name__ == "__main__":
    main()
