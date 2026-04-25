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
pip install -U transformers accelerate bitsandbytes pillow sentencepiece tqdm
```

Recommended T4 run:

```bash
chmod +x run_qwen3vl.sh
./run_qwen3vl.sh data.jsonl scored.jsonl .
```

Run directly on ViTextVQA JSON:

```bash
./run_qwen3vl.sh ViTextVQA_train.json scored.jsonl ViTextVQA_images/st_images
```

If memory is stable, try:

```bash
./run_qwen3vl.sh data.jsonl scored.jsonl . --batch-size 2 --short-side 640 --long-side 1024 --max-pixels 655360
```
