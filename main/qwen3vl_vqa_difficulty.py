import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration


SYSTEM_PROMPT = (
    "You are an expert evaluator for Vietnamese scene-text Visual Question Answering "
    "(VQA) datasets. Your role is to assess the difficulty of each (image, question, "
    "answer) triplet for training an OCR-VQA model. You understand Vietnamese language, "
    "typography, and scene-text challenges. You must respond ONLY in valid JSON. "
    "No explanation outside the JSON block."
)


def build_scoring_prompt(question: str, answer: str) -> str:
    return f"""Evaluate the difficulty of this VQA sample for an OCR model.
Analyze the image carefully.

Question: {question}
Ground-truth Answer: {answer}

Score each dimension from 1 (easiest) to 5 (hardest):
1. text_visibility — How clear/legible is the text in the image? (1=large, clear, high-contrast | 5=blurry, small, low-contrast)
2. text_orientation — How complex is the text layout/orientation? (1=horizontal, single line | 5=rotated, curved, multi-direction)
3. text_density — How much text is present relative to image area? (1=minimal text | 5=dense text regions, many words)
4. linguistic_complexity — How hard is the question/answer linguistically? Consider Vietnamese morphology, rare vocab, abbreviations. (1=simple common words | 5=technical terms, abbreviations, mixed-script)
5. reasoning_required — Does answering require reading + reasoning beyond simple extraction? (1=direct text extraction | 5=inference, calculation, multi-step reasoning)
6. ocr_ambiguity — How likely is character-level confusion? Consider Vietnamese diacritics (ắ/ặ/ấ/ầ), similar glyphs (0/O, l/1/I). (1=no ambiguous characters | 5=many confusable diacritics/glyphs)

Return ONLY this JSON:
{{
  "scores": {{
    "text_visibility": <1-5>,
    "text_orientation": <1-5>,
    "text_density": <1-5>,
    "linguistic_complexity": <1-5>,
    "reasoning_required": <1-5>,
    "ocr_ambiguity": <1-5>
  }},
  "weighted_difficulty": <float 1.0-5.0>,
  "difficulty_tier": <"easy" | "medium" | "hard" | "very_hard">,
  "rationale": "<2 sentences max explaining the dominant difficulty factors>"
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
    parser.add_argument("--image-key", default="image", help="Image path field name.")
    parser.add_argument("--question-key", default="question", help="Question field name.")
    parser.add_argument("--answer-key", default="answer", help="Answer field name.")
    parser.add_argument(
        "--id-key",
        default="id",
        help="Optional id field used for resume and output tracking.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        choices=[1, 2],
        help="Keep this small to avoid VRAM spikes on T4.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Token budget for JSON output. Use >=192 to avoid truncation.",
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
        help="Extra safety cap after downscaling to bound the vision encoder cost.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="T4 should use float16. bf16 is included for newer GPUs.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "eager"],
        help="Use sdpa by default; flash attention is not assumed on T4.",
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
        "--limit",
        type=int,
        default=None,
        help="Optional cap for quick tests.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Flush output every N samples so long runs are resumable.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print throughput every N samples.",
    )
    return parser.parse_args()


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


def load_model_and_processor(
    args: argparse.Namespace,
) -> Tuple[Qwen3VLForConditionalGeneration, Any]:
    quantization_config = build_quant_config(args)
    torch_dtype = resolve_dtype(args.dtype)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.model)
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
            sample_id = row.get(id_key)
            if sample_id is not None and not row.get("error"):
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

        # ViTextVQA exports sometimes use image_id as an index into images.
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


def build_messages(question: str, answer: str) -> List[Dict[str, Any]]:
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


def chunked(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def infer_input_device(model: Qwen3VLForConditionalGeneration) -> torch.device:
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


def parse_model_json(text: str) -> Dict[str, Any]:
    candidate = extract_json_candidate(text)
    return json.loads(candidate)


def build_output_record(
    sample: Dict[str, Any],
    score: Optional[Dict[str, Any]],
    raw_output: str,
    error: Optional[str],
) -> Dict[str, Any]:
    result = dict(sample)
    result["model_output"] = raw_output
    if score is not None:
        result["difficulty"] = score
    if error is not None:
        result["error"] = error
    return result


def is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message


def run_batch_generation(
    model: Qwen3VLForConditionalGeneration,
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


def should_retry_json_parse(raw_output: str, error: Exception) -> bool:
    message = str(error)
    if "balanced JSON object" not in message and "Expecting" not in message:
        return False

    stripped = raw_output.strip()
    if not stripped:
        return True

    return stripped.count("{") > stripped.count("}") or not stripped.endswith("}")


def main() -> None:
    args = parse_args()
    setup_logging()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. A T4 is expected here.")

    torch.set_grad_enabled(False)
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    done_ids = load_done_ids(args.output, args.id_key)
    all_samples = []
    for sample in iter_input_samples(args.input):
        sample_id = sample.get(args.id_key)
        if sample_id is not None and str(sample_id) in done_ids:
            continue
        all_samples.append(sample)
        if args.limit is not None and len(all_samples) >= args.limit:
            break

    if not all_samples:
        logging.info("No pending samples found.")
        return

    model, processor = load_model_and_processor(args)
    input_device = infer_input_device(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Pending samples: %d", len(all_samples))
    logging.info("Model: %s", args.model)
    logging.info(
        "Config: 4bit=%s batch_size=%d max_new_tokens=%d short_side=%d long_side=%d max_pixels=%d",
        not args.disable_4bit,
        args.batch_size,
        args.max_new_tokens,
        args.short_side,
        args.long_side,
        args.max_pixels,
    )

    started_at = time.time()
    written = 0
    error_rows = 0

    with args.output.open("a", encoding="utf-8") as out_handle:
        with tqdm(total=len(all_samples), desc="Scoring", unit="sample") as progress:
            for batch_index, batch in enumerate(chunked(all_samples, args.batch_size), start=1):
                pil_images = []
                texts = []
                for sample in batch:
                    image_path = resolve_image_path(sample[args.image_key], args.image_root)
                    pil_images.append(
                        load_image(
                            image_path,
                            short_side_limit=args.short_side,
                            long_side_limit=args.long_side,
                            max_pixels=args.max_pixels,
                        )
                    )
                    messages = build_messages(
                        question=str(sample[args.question_key]),
                        answer=str(sample[args.answer_key]),
                    )
                    texts.append(
                        processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )

                model_inputs = prepare_model_inputs(
                    processor=processor,
                    texts=texts,
                    images=pil_images,
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
                    for single_sample, single_image, single_text in zip(batch, pil_images, texts):
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

                for sample, raw_output, sample_image, sample_text in zip(
                    batch,
                    outputs,
                    pil_images,
                    texts,
                ):
                    error = None
                    parsed = None
                    try:
                        parsed = parse_model_json(raw_output)
                    except Exception as exc:
                        error = str(exc)
                        if should_retry_json_parse(raw_output, exc):
                            retry_tokens = max(args.max_new_tokens * 2, 192)
                            logging.warning(
                                "Retrying truncated/invalid JSON for sample id=%s with max_new_tokens=%d",
                                sample.get(args.id_key),
                                retry_tokens,
                            )
                            try:
                                retry_inputs = prepare_model_inputs(
                                    processor=processor,
                                    texts=[sample_text],
                                    images=[sample_image],
                                    input_device=input_device,
                                )
                                retry_output = run_batch_generation(
                                    model=model,
                                    processor=processor,
                                    model_inputs=retry_inputs,
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

                for image in pil_images:
                    image.close()

    elapsed = max(time.time() - started_at, 1e-6)
    logging.info(
        "Finished %d samples in %.2f minutes (%.2f samples/s)",
        written,
        elapsed / 60.0,
        written / elapsed,
    )


if __name__ == "__main__":
    main()
