# Project Proposal — Text-to-Image Generation with CLIP-Guided Latent Diffusion

**Team:** [Member A], [Member B], [Member C], [Member D] · **Date:** November 2026 · **Repo:** `github.com/<org>/text2image-diffusion`

## 1. Problem & Goal
Generate images that faithfully match natural-language descriptions. We'll start from a pretrained latent diffusion model, measure how well its outputs match their prompts on everyday scenes, then improve that match by fine-tuning on a curated image–caption subset. **Success metric:** a higher CLIPScore and better text→image retrieval than the zero-shot baseline, plus a human preference study on held-out prompts.

## 2. Dataset — MS-COCO 2017 Captions (curated subset)
| | |
|---|---|
| **Source** | COCO val2017: 5,000 images, ~25k human captions (5 per image) |
| **Why COCO over Flickr30k** | More object/scene variety (80 categories), and it's the standard benchmark for text-to-image FID/CLIPScore. The 1 GB val split fits our 1k–5k target without the 18 GB train download. |
| **Cleaning** | Normalize whitespace and punctuation; keep captions of 5–40 words that fit in 77 CLIP tokens and are ASCII; drop exact duplicates; drop images whose short side is under 256 px or whose aspect ratio is over 2:1; fix EXIF orientation and convert to RGB |
| **Preprocessing** | Resize the short side to 512 px and center-crop to 512×512. Keep one primary caption per image and the rest for augmentation. 90/10 train/val split (seed 42). Output is HF-`imagefolder` `metadata.jsonl` |
| **Size** | ≈4.5k–5k pairs after filtering (exact counts in `data/processed/stats.json`) |

## 3. Model & Method
- **Text encoder:** CLIP ViT-L/14. Each prompt becomes a 77×768 per-token embedding (fed to the UNet through cross-attention) and a pooled 768-d embedding (used for scoring and retrieval).
- **Generator:** Stable Diffusion v1.5 (latent diffusion: VAE plus a UNet conditioned on text), fp16, DPM-Solver++ with 30 steps and CFG 7.5. It fits our 8 GB RTX 5060 GPUs.
- **Phase 1 (this milestone):** zero-shot baseline. CLIP Recall@K and CLIPScore on real pairs, plus 5 sample prompt→image runs.
- **Phase 2:** LoRA fine-tuning of the UNet's cross-attention layers on the training split (rank 4–16, about 3M trainable parameters).
- **Phase 3:** ablations on guidance scale, step count, and caption choice (primary vs. augmented), then a comparison with a CLIP-guided or prompt-weighting variant.
- **Evaluation:** CLIPScore, FID on the 500-image val split (`torchmetrics`), text→image Recall@K, and a small blind human preference study.

## 4. Roles
| Member | Role | Owns |
|---|---|---|
| [Member A] | **Data lead** | Download, cleaning, and filtering scripts; dataset stats and EDA |
| [Member B] | **Modeling lead** | Diffusers pipeline, LoRA fine-tuning, GPU/experiment configs |
| [Member C] | **Evaluation lead** | CLIPScore, FID, and Recall@K harness; human study; result plots |
| [Member D] | **Integration / PM** | Repo, environment, CI, notebooks, milestone reports, demo |

## 5. Milestone 1 Deliverables (this submission)
1. Dataset chosen and justified (COCO val2017 subset)
2. `scripts/preprocess.py` cleans and builds 1k–5k pairs
3. Environment: PyTorch (CUDA 12.8), Diffusers, and Transformers/CLIP, checked with `scripts/check_env.py`
4. Baseline notebook `notebooks/01_text_to_embedding_baseline.ipynb`
5. Five sample prompt→image runs: `outputs/samples/grid.png` and `runs.json`
6. GitHub repo initialized

**Risks:** limited VRAM (mitigated with fp16, attention slicing, and LoRA instead of full fine-tuning); COCO caption noise (mitigated by filtering and choosing the best caption per image); dataset licensing (COCO images are under Flickr CC licenses, so we use them for research only).
