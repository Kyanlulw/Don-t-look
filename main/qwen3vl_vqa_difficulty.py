"""
qwen3vl_vqa_difficulty_fast.py  —  speed-optimised rewrite
=========================================================
Key changes vs the original:
  1. Shorter system prompt  (~15 tok  vs ~60 tok)
  2. Shorter scoring prompt (~90 tok  vs ~350 tok) — removed verbose
     dimension descriptions; removed 'rationale' (biggest output-token sink)
  3. Qwen3 thinking disabled  (enable_thinking=False in apply_chat_template)
     — original silently generated a full <think>…</think> block before JSON
  4. Smaller image cap  (short=336 long=448 max_pixels=150 528)
     → ~192 visual tokens  vs  ~1 250  =  6× fewer vision tokens
  5. Batch size default raised to 4 (fits comfortably with smaller images)
  6. max_new_tokens default lowered to 60 (no rationale needed)
  7. Minor: flush / log every 200 samples, reduce I/O pressure

Expected throughput on a Kaggle T4 (16 GB):
  original  ~230 samples / hour   → 35 k in ~150 hours
  optimised ~2 000-4 000 / hour   → 5 k in   1-3 hours
"""

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


# ── Prompts ──────────────────────────────────────────────────────────────────
# CHANGE 1 & 2: Drastically shorter prompts.
#   • Removed multi-line dimension descriptions (reader already sees the image)
#   • Removed 'rationale' from the output schema (saves 30-80 output tokens per
#     sample and eliminates most truncation-retry events)
#   • Kept identical field names so downstream code needs no changes

SYSTEM_PROMPT = (
    "Assess Vietnamese OCR-VQA difficulty. Return one JSON object only."
)


def build_scoring_prompt(question: str, answer: str) -> str:
    """Build a compact scoring prompt (~90 tokens vs the original ~350)."""
    return (
        f"Vietnamese scene-text VQA sample.\n"
        f"Q: {question}\n"
        f"A: {answer}\n\n"
        "Rate each dimension 1(easiest)–5(hardest):\n"
        "  text_visibility   – legibility / contrast\n"
        "  text_orientation  – layout / rotation complexity\n"
        "  text_density      – amount of text in image\n"
        "  linguistic_complexity – Vietnamese vocab / abbreviations\n"
        "  reasoning_required    – inference beyond simple extraction\n"
        "  ocr_ambiguity     – diacritic / glyph confusion (ắ/ặ, 0/O …)\n\n"
        "Output ONLY valid JSON, exactly this shape:\n"
        '{"scores":{"text_visibility":?,"text_orientation":?,'
        '"text_density":?,"linguistic_complexity":?,'
        '"reasoning_required":?,"ocr_ambiguity":?},'
        '"weighted_difficulty":?,'
        '"difficulty_tier":"easy|medium|hard|very_hard"}'
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score OCR-VQA difficulty with Qwen3-VL (speed-optimised)."
    )
    parser.add_argument("--input",  type=Path, required=True,
                        help="Input .jsonl or ViTextVQA-style .json.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL file.")
    parser.add_argument("--model",  default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Hugging Face model id.")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--image-key",    default="image")
    parser.add_argument("--question-key", default="question")
    parser.add_argument("--answer-key",   default="answer")
    parser.add_argument("--id-key",       default="id")

    # CHANGE 5: default batch size raised from 1 → 4
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Samples per forward pass. 4 fits T4 with small images.")

    # CHANGE 6: default max_new_tokens lowered from 96 → 60 (no rationale)
    parser.add_argument("--max-new-tokens", type=int, default=60,
                        help="Token budget for JSON output (no rationale = ~55 tok).")

    # CHANGE 4: tighter image caps → far fewer visual tokens
    parser.add_argument("--short-side", type=int, default=336,
                        help="Downscale short side to this (was 768).")
    parser.add_argument("--long-side",  type=int, default=448,
                        help="Downscale long side to this (was 1280).")
    parser.add_argument("--max-pixels", type=int, default=336 * 448,
                        help="Hard pixel cap after downscaling (~150 k, was ~983 k).")

    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--attn-implementation", default="sdpa",
                        choices=["sdpa", "eager"])
    parser.add_argument("--disable-4bit", action="store_true")
    parser.add_argument("--double-quant", action="store_true")

    # CHANGE 3: thinking is disabled by default; pass --enable-thinking to turn on
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Allow Qwen3 chain-of-thought (slow; off by default).")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=200)
    parser.add_argument("--log-every",   type=int, default=200)
    return parser.parse_args()


# ── Logging / utils ──────────────────────────────────────────────────────────
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def build_quant_config(args: argparse.Namespace) -> Optional[BitsAndBytesConfig]:
    if args.disable_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=resolve_dtype(args.dtype),
        bnb_4bit_use_double_quant=args.double_quant,
    )


def load_model_and_processor(
    args: argparse.Namespace,
) -> Tuple[Qwen3VLForConditionalGeneration, Any]:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=build_quant_config(args),
        torch_dtype=resolve_dtype(args.dtype),
        device_map="auto",
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        # CHANGE 4 (processor side): tell the processor's image encoder the
        # same pixel cap so it won't up-sample internally.
        min_pixels=28 * 28,
        max_pixels=args.max_pixels,
    )
    model.eval()
    return model, processor


# ── I/O helpers ──────────────────────────────────────────────────────────────
def load_done_ids(output_path: Path, id_key: str) -> set:
    done: set = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = row.get(id_key)
            if sid is not None and not row.get("error"):
                done.add(str(sid))
    return done


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {lineno} of {path}") from exc


def iter_vitextvqa_json(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("Expected top-level 'images' and 'annotations' lists.")

    images_by_id = {img["id"]: img for img in images if isinstance(img, dict) and "id" in img}

    for idx, ann in enumerate(annotations, 1):
        if not isinstance(ann, dict):
            continue
        image_id = ann.get("image_id")
        image_entry = images_by_id.get(image_id)
        if image_entry is None and isinstance(image_id, int) and 0 <= image_id < len(images):
            maybe = images[image_id]
            if isinstance(maybe, dict):
                image_entry = maybe
        if image_entry is None:
            raise ValueError(f"Cannot resolve image for annotation {idx} (image_id={image_id}).")

        image_value = image_entry.get("filename") or image_entry.get("image")
        if not image_value:
            raise ValueError(f"Missing image filename for annotation {idx}.")

        answers = ann.get("answers")
        answer_value = answers[0] if isinstance(answers, list) and answers else ann.get("answer", "")

        yield {
            "id": ann.get("id", idx),
            "image": str(image_value),
            "question": str(ann.get("question", "")),
            "answer": str(answer_value),
        }


def iter_input_samples(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        yield from iter_jsonl(path)
    elif path.suffix.lower() == ".json":
        yield from iter_vitextvqa_json(path)
    else:
        raise ValueError(f"Unsupported format: {path}. Use .jsonl or .json")


def resolve_image_path(raw_path: str, image_root: Optional[Path]) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if image_root is not None:
        return image_root / path
    return path


# ── Image loading ─────────────────────────────────────────────────────────────
def maybe_resize_image(
    image: Image.Image,
    short_side_limit: int,
    long_side_limit: int,
    max_pixels: int,
) -> Image.Image:
    w, h = image.size
    scale = 1.0
    short_side = min(w, h)
    long_side  = max(w, h)
    if short_side > short_side_limit:
        scale = min(scale, short_side_limit / short_side)
    if long_side > long_side_limit:
        scale = min(scale, long_side_limit / long_side)
    if w * h > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / float(w * h)))
    if scale >= 1.0:
        return image
    new_w = max(28, int(round(w * scale)))
    new_h = max(28, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def load_image(path: Path, short_side_limit: int, long_side_limit: int,
               max_pixels: int) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        return maybe_resize_image(img, short_side_limit, long_side_limit, max_pixels)


# ── Message building ──────────────────────────────────────────────────────────
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


def build_scoring_prompt(question: str, answer: str) -> str:
    """Override with a smaller prompt to cut decode time on T4."""
    return (
        f"Q:{question}\n"
        f"A:{answer}\n"
        "Score image difficulty 1-5.\n"
        "v=visibility,o=orientation,d=density,l=language,r=reasoning,a=ambiguity.\n"
        'Return exactly: {"s":{"v":1,"o":1,"d":1,"l":1,"r":1,"a":1},"wd":1.0}'
    )


# ── JSON parsing ──────────────────────────────────────────────────────────────
def extract_json_candidate(text: str) -> str:
    stripped = text.strip()
    # Strip markdown fences if present
    if stripped.startswith("```"):
        lines = [l for l in stripped.splitlines() if not l.strip().startswith("```")]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth, in_string, escaped = 0, False, False
    for idx, ch in enumerate(stripped[start:], start=start):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start: idx + 1]
    raise ValueError("Could not find a balanced JSON object in model output.")


def parse_model_json(text: str) -> Dict[str, Any]:
    return json.loads(extract_json_candidate(text))


def should_retry_json_parse(raw_output: str, error: Exception) -> bool:
    msg = str(error)
    if "balanced JSON object" not in msg and "Expecting" not in msg:
        return False
    stripped = raw_output.strip()
    if not stripped:
        return True
    return stripped.count("{") > stripped.count("}") or not stripped.endswith("}")


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


def parse_model_json(text: str) -> Dict[str, Any]:
    payload = json.loads(extract_json_candidate(text))
    raw_scores = payload.get("s", payload.get("scores", {}))
    aliases = {
        "v": "text_visibility",
        "text_visibility": "text_visibility",
        "o": "text_orientation",
        "text_orientation": "text_orientation",
        "d": "text_density",
        "text_density": "text_density",
        "l": "linguistic_complexity",
        "linguistic_complexity": "linguistic_complexity",
        "r": "reasoning_required",
        "reasoning_required": "reasoning_required",
        "a": "ocr_ambiguity",
        "ocr_ambiguity": "ocr_ambiguity",
    }

    scores: Dict[str, int] = {}
    for key, value in raw_scores.items():
        full_key = aliases.get(key)
        if full_key is None:
            continue
        scores[full_key] = max(1, min(5, int(round(float(value)))))

    if not scores:
        raise ValueError("Missing difficulty scores in model output.")

    weighted_difficulty = payload.get("wd", payload.get("weighted_difficulty"))
    if weighted_difficulty is None:
        weighted_difficulty = sum(scores.values()) / len(scores)
    weighted_difficulty = round(max(1.0, min(5.0, float(weighted_difficulty))), 2)

    if weighted_difficulty < 2.0:
        difficulty_tier = "easy"
    elif weighted_difficulty < 3.0:
        difficulty_tier = "medium"
    elif weighted_difficulty < 4.0:
        difficulty_tier = "hard"
    else:
        difficulty_tier = "very_hard"

    return {
        "scores": scores,
        "weighted_difficulty": weighted_difficulty,
        "difficulty_tier": difficulty_tier,
    }


# ── Model inference ───────────────────────────────────────────────────────────
def chunked(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i: i + size]


def infer_input_device(model: Qwen3VLForConditionalGeneration) -> torch.device:
    if hasattr(model, "device") and str(model.device) != "meta":
        return model.device
    return next(model.parameters()).device


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
        k: v.to(input_device) if hasattr(v, "to") else v
        for k, v in model_inputs.items()
    }


def is_oom_error(exc: Exception) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def run_batch_generation(
    model: Qwen3VLForConditionalGeneration,
    processor: Any,
    model_inputs: Dict[str, Any],
    max_new_tokens: int,
) -> List[str]:
    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
        )
    prompt_len = model_inputs["input_ids"].shape[1]
    return processor.batch_decode(
        generated_ids[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    setup_logging()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.set_grad_enabled(False)
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    done_ids = load_done_ids(args.output, args.id_key)
    all_samples: List[Dict[str, Any]] = []
    for sample in iter_input_samples(args.input):
        sid = sample.get(args.id_key)
        if sid is not None and str(sid) in done_ids:
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

    logging.info("Pending samples : %d", len(all_samples))
    logging.info("Model           : %s", args.model)
    logging.info(
        "Config: 4bit=%s batch=%d max_new_tokens=%d "
        "short=%d long=%d max_pixels=%d thinking=%s",
        not args.disable_4bit,
        args.batch_size,
        args.max_new_tokens,
        args.short_side,
        args.long_side,
        args.max_pixels,
        args.enable_thinking,
    )

    started_at = time.time()
    written = error_rows = 0

    # CHANGE 3: build thinking flag once for apply_chat_template
    thinking_kwargs: Dict[str, Any] = {}
    try:
        # Qwen3 processors expose enable_thinking in apply_chat_template
        import inspect
        sig = inspect.signature(processor.apply_chat_template)
        if "enable_thinking" in sig.parameters:
            thinking_kwargs["enable_thinking"] = args.enable_thinking
    except Exception:
        pass  # older processors: silently skip

    with args.output.open("a", encoding="utf-8") as out_fh:
        with tqdm(total=len(all_samples), desc="Scoring", unit="sample") as bar:
            for batch_idx, batch in enumerate(chunked(all_samples, args.batch_size), 1):
                pil_images: List[Image.Image] = []
                texts: List[str] = []

                for sample in batch:
                    img_path = resolve_image_path(sample[args.image_key], args.image_root)
                    pil_images.append(
                        load_image(
                            img_path,
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
                            **thinking_kwargs,          # CHANGE 3
                        )
                    )

                model_inputs = prepare_model_inputs(processor, texts, pil_images, input_device)

                try:
                    outputs = run_batch_generation(model, processor, model_inputs, args.max_new_tokens)
                except Exception as exc:
                    if len(batch) == 1 or not is_oom_error(exc):
                        raise
                    logging.warning("OOM on batch %d (size %d). Retrying one-by-one.", batch_idx, len(batch))
                    torch.cuda.empty_cache()
                    outputs = []
                    for single_sample, single_img, single_text in zip(batch, pil_images, texts):
                        single_inputs = prepare_model_inputs(
                            processor, [single_text], [single_img], input_device
                        )
                        try:
                            out = run_batch_generation(
                                model, processor, single_inputs, args.max_new_tokens
                            )[0]
                        except Exception as exc2:
                            if is_oom_error(exc2):
                                torch.cuda.empty_cache()
                                out = json.dumps({"error": "cuda_oom_during_generation"}, ensure_ascii=False)
                            else:
                                raise
                        outputs.append(out)

                for sample, raw_output, pil_img, sample_text in zip(batch, outputs, pil_images, texts):
                    error: Optional[str] = None
                    parsed: Optional[Dict[str, Any]] = None
                    try:
                        parsed = parse_model_json(raw_output)
                    except Exception as exc:
                        error = str(exc)
                        if should_retry_json_parse(raw_output, exc):
                            retry_tokens = max(args.max_new_tokens * 2, 120)
                            logging.warning(
                                "Retrying truncated JSON for id=%s with max_new_tokens=%d",
                                sample.get(args.id_key), retry_tokens,
                            )
                            try:
                                retry_inputs = prepare_model_inputs(
                                    processor, [sample_text], [pil_img], input_device
                                )
                                retry_out = run_batch_generation(
                                    model, processor, retry_inputs, retry_tokens
                                )[0]
                                parsed = parse_model_json(retry_out)
                                raw_output = retry_out
                                error = None
                            except Exception as retry_exc:
                                error = f"{error} | retry_failed: {retry_exc}"

                    if error is not None:
                        error_rows += 1

                    record = build_output_record(sample, parsed, raw_output, error)
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    bar.update(1)

                if written % args.flush_every == 0:
                    out_fh.flush()

                if written % args.log_every == 0:
                    elapsed = max(time.time() - started_at, 1e-6)
                    speed = written / elapsed
                    bar.set_postfix(errors=error_rows, sps=f"{speed:.1f}")
                    logging.info(
                        "Processed %d/%d | %.1f samples/s | eta %.0f min",
                        written,
                        len(all_samples),
                        speed,
                        (len(all_samples) - written) / speed / 60,
                    )

                for img in pil_images:
                    img.close()

    elapsed = max(time.time() - started_at, 1e-6)
    logging.info(
        "Done: %d samples in %.1f min (%.1f samples/s)",
        written, elapsed / 60, written / elapsed,
    )


if __name__ == "__main__":
    main()
