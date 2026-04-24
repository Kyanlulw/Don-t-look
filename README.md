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

```powershell
python qwen3vl_vqa_difficulty.py `
  --input data.jsonl `
  --output scored.jsonl `
  --image-root . `
  --model Qwen/Qwen3-VL-4B-Instruct `
  --batch-size 1 `
  --max-new-tokens 96 `
  --short-side 768 `
  --long-side 1280 `
  --max-pixels 983040 `
  --double-quant
```

If memory is stable, try:

```powershell
python qwen3vl_vqa_difficulty.py `
  --input data.jsonl `
  --output scored.jsonl `
  --image-root . `
  --batch-size 2 `
  --short-side 640 `
  --long-side 1024 `
  --max-pixels 655360 `
  --double-quant
```
