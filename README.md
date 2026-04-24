# Qwen3-VL OCR-VQA Difficulty Scoring

Input format is JSONL with at least these fields:

```json
{"id":"sample-0001","image":"images/example.jpg","question":"Biển hiệu ghi gì?","answer":"NHA KHOA"}
```

Recommended install:

```powershell
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U transformers accelerate bitsandbytes pillow sentencepiece
```

Recommended T4 run:

```bash
chmod +x run_qwen3vl.sh
./run_qwen3vl.sh data.jsonl scored.jsonl .
```

If memory is stable, try:

```bash
./run_qwen3vl.sh data.jsonl scored.jsonl . \
  --batch-size 2 \
  --short-side 640 \
  --long-side 1024 \
  --max-pixels 655360
```
