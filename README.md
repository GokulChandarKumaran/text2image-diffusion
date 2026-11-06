# text2image-diffusion

Text-to-image generation with CLIP text embeddings and Stable Diffusion, trained and evaluated on a curated MS-COCO caption subset.
See [docs/proposal.md](docs/proposal.md) for the full proposal.

## Layout
```
scripts/
  check_env.py          # verify torch / CUDA / diffusers / CLIP
  download_coco.py      # COCO 2017 val images + captions -> data/raw/
  preprocess.py         # clean, filter, resize -> data/processed/ (metadata.jsonl, stats.json)
  generate_samples.py   # 5 prompt->image runs + CLIPScore -> outputs/samples/
notebooks/
  01_text_to_embedding_baseline.ipynb   # tokenization, 77x768 conditioning, Recall@K, CLIPScore
docs/proposal.md
```

## Setup
Use Python 3.11 or 3.12. PyTorch wheels may not be available yet for newer versions.
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 50-series needs cu128
pip install -r requirements.txt
python scripts/check_env.py
```

## Run Milestone 1
```bash
python scripts/download_coco.py           # ~1 GB
python scripts/preprocess.py --n 5000     # -> ~5k clean pairs
jupyter notebook notebooks/01_text_to_embedding_baseline.ipynb
python scripts/generate_samples.py        # ~4 GB model download on first run
```

## Milestone 1 checklist
- [x] Dataset chosen: COCO val2017 (curated subset)
- [x] Cleaning and preprocessing pipeline for 1k–5k pairs
- [x] Environment: PyTorch, Diffusers, and CLIP
- [x] Baseline text→embedding notebook
- [ ] 5 sample prompt→image runs (`outputs/samples/grid.png`)
- [x] Repo initialized
