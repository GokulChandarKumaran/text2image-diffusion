"""Sample 5 prompt -> image test runs with Stable Diffusion 1.5 (fits an 8 GB GPU in fp16).

Each run uses a fixed seed for reproducibility and is scored with CLIPScore
(CLIP ViT-L/14 cosine x 100). Outputs:
  outputs/samples/run_{i}.png   individual images
  outputs/samples/grid.png      1x5 grid with prompts (commit this)
  outputs/samples/runs.json     prompt, seed, steps, guidance, time, CLIPScore (commit this)

    python scripts/generate_samples.py
    python scripts/generate_samples.py --steps 30 --guidance 7.5 --model stable-diffusion-v1-5/stable-diffusion-v1-5
"""
import argparse
import json
import textwrap
import time
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "samples"

# Mix of COCO-style captions (in-distribution) and harder compositional prompts.
PROMPTS = [
    "A man riding a wave on top of a surfboard.",
    "A plate of food with broccoli, rice and grilled chicken on a wooden table.",
    "A red double-decker bus driving down a rainy city street at night.",
    "Two cats sleeping on a blue couch next to a remote control.",
    "A giraffe standing next to a small wooden house in a snowy field.",
]
NEGATIVE = "blurry, low quality, distorted, deformed, watermark, text"


@torch.no_grad()
def clip_score(model, processor, image, text, device):
    inp = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
    t = torch.nn.functional.normalize(model.get_text_features(input_ids=inp.input_ids, attention_mask=inp.attention_mask), dim=-1)
    i = torch.nn.functional.normalize(model.get_image_features(pixel_values=inp.pixel_values), dim=-1)
    return float(100 * (t * i).sum().clamp(min=0))


def make_grid(images, prompts, size=384):
    pad = 70
    grid = Image.new("RGB", (size * len(images), size + pad), "white")
    draw = ImageDraw.Draw(grid)
    for k, (im, p) in enumerate(zip(images, prompts)):
        grid.paste(im.resize((size, size)), (k * size, 0))
        draw.multiline_text((k * size + 6, size + 4), "\n".join(textwrap.wrap(p, 48)[:4]), fill="black")
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    OUT.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # keeps peak VRAM well under 8 GB

    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype).to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    runs, images = [], []
    for i, prompt in enumerate(PROMPTS):
        seed = args.seed + i
        g = torch.Generator(device=device).manual_seed(seed)
        t0 = time.perf_counter()
        img = pipe(prompt, negative_prompt=NEGATIVE, num_inference_steps=args.steps,
                   guidance_scale=args.guidance, height=args.size, width=args.size, generator=g).images[0]
        dt = time.perf_counter() - t0
        score = clip_score(clip, proc, img, prompt, device)
        img.save(OUT / f"run_{i + 1}.png")
        images.append(img)
        runs.append({"run": i + 1, "prompt": prompt, "seed": seed, "steps": args.steps,
                     "guidance": args.guidance, "seconds": round(dt, 2), "clip_score": round(score, 2)})
        print(f"[{i + 1}/5] {dt:5.1f}s  CLIPScore={score:5.2f}  {prompt}")

    make_grid(images, PROMPTS).save(OUT / "grid.png")
    summary = {"model": args.model, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
               "mean_clip_score": round(sum(r["clip_score"] for r in runs) / len(runs), 2), "runs": runs}
    (OUT / "runs.json").write_text(json.dumps(summary, indent=2))
    print(f"Mean CLIPScore: {summary['mean_clip_score']}  -> {OUT}")


if __name__ == "__main__":
    main()
