# Qwen3-VL OCR-VQA Difficulty Scoring

Input format supports either:

- JSONL with at least these fields:

```json
{"id":"sample-0001","image":"images/example.jpg","question":"Biển hiệu ghi gì?","answer":"NHA KHOA"}
```

- ViTextVQA-style JSON (top-level `images` and `annotations`). The script automatically maps each annotation to `{id, image, question, answer}` and uses the first item in `answers`.

Recommended install:

```powershell
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U transformers accelerate bitsandbytes pillow sentencepiece tqdm einops timm
```

Recommended T4 run:

```bash
chmod +x run_qwen3vl.sh
./run_qwen3vl.sh data.jsonl scored.jsonl .
```

By default, `run_qwen3vl.sh` runs with a fixed seed and a small sample limit.
To run a different number, pass `--limit N`.
Example:

```bash
./run_qwen3vl.sh data.jsonl scored.jsonl . --limit 500
```

Seeded sampling (deterministic random picks/mix order):

```bash
./run_qwen3vl.sh data.jsonl scored.jsonl . --seed 123
```

Run directly on ViTextVQA JSON:

```bash
./run_qwen3vl.sh ViTextVQA_train.json scored.jsonl ViTextVQA_images/st_images
```

Mix two datasets with per-dataset ranges and random picks:

```bash
./run_qwen3vl.sh data_a.jsonl mixed_scored.jsonl . --input-2 data_b.jsonl --image-root-2 /path/to/images_b --range-1 0:5000 --range-2 1000:7000 --pick-1 800 --pick-2 800 --seed 123
```

Notes:

- `--range-1` / `--range-2` use `start:end` (0-based, end exclusive).
- `--pick-1` / `--pick-2` sample randomly within each ranged subset.
- When `--input-2` is provided, samples from both datasets are shuffled together using `--seed`.

If memory is stable, try:

```bash
./run_qwen3vl.sh data.jsonl scored.jsonl . --batch-size 2 --short-side 640 --long-side 1024 --max-pixels 655360
```
