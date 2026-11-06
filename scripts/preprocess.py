"""Clean & preprocess COCO val2017 into a curated image-caption subset.

Pipeline
  1. Load captions_val2017.json (5 captions / image).
  2. Clean captions: strip, collapse whitespace, fix spacing before punctuation,
     sentence-case, ensure trailing period.
  3. Filter captions: 5-40 words, <=77 CLIP tokens, ASCII-only, no exact dups.
  4. Filter images: RGB-convertible, short side >= 256 px, aspect ratio <= 2:1.
  5. Pick the best caption per image (longest that still fits 77 tokens) -> 1 pair/image,
     but keep all valid captions in `all_captions` for augmentation.
  6. Resize (short side -> 512) + center-crop 512x512, save as JPEG.
  7. Write metadata.jsonl (HF `imagefolder` compatible) + train/val split + stats.

    python scripts/preprocess.py --n 5000 --size 512
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import CLIPTokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def clean_caption(c: str) -> str:
    c = c.strip().replace("\n", " ")
    c = re.sub(r"\s+", " ", c)
    c = re.sub(r"\s+([.,!?;:])", r"\1", c)
    c = c.rstrip(" .")
    if not c:
        return ""
    return c[0].upper() + c[1:] + "."


def valid_caption(c: str, tok: CLIPTokenizer, max_tokens: int = 77) -> bool:
    words = c.split()
    if not (5 <= len(words) <= 40) or not c.isascii():
        return False
    return len(tok(c).input_ids) <= max_tokens


def load_image(path: Path, min_side: int, max_ratio: float):
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        return None
    w, h = im.size
    if min(w, h) < min_side or max(w, h) / min(w, h) > max_ratio:
        return None
    return im


def resize_center_crop(im: Image.Image, size: int) -> Image.Image:
    return ImageOps.fit(im, (size, size), method=Image.BICUBIC, centering=(0.5, 0.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="max pairs to keep (1k-5k)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--min-side", type=int, default=256)
    ap.add_argument("--max-ratio", type=float, default=2.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ann = json.loads((RAW / "annotations" / "captions_val2017.json").read_text())
    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    stats = defaultdict(int)
    caps = defaultdict(list)
    seen = set()
    for a in ann["annotations"]:
        stats["captions_raw"] += 1
        c = clean_caption(a["caption"])
        key = c.lower()
        if key in seen:
            stats["drop_duplicate"] += 1
            continue
        if not valid_caption(c, tok):
            stats["drop_caption_filter"] += 1
            continue
        seen.add(key)
        caps[a["image_id"]].append(c)

    ids = sorted(caps)
    random.Random(args.seed).shuffle(ids)

    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for iid in tqdm(ids, desc="images"):
        if len(rows) >= args.n:
            break
        im = load_image(RAW / "val2017" / id2file[iid], args.min_side, args.max_ratio)
        if im is None:
            stats["drop_image_filter"] += 1
            continue
        fname = f"{iid:012d}.jpg"
        resize_center_crop(im, args.size).save(img_dir / fname, quality=95)
        best = max(caps[iid], key=len)
        rows.append({"file_name": f"images/{fname}", "image_id": iid, "text": best, "all_captions": caps[iid]})

    n_val = int(len(rows) * args.val_frac)
    for i, r in enumerate(rows):
        r["split"] = "val" if i < n_val else "train"

    with open(OUT / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    lengths = [len(r["text"].split()) for r in rows]
    stats.update(
        pairs_kept=len(rows),
        train=len(rows) - n_val,
        val=n_val,
        caption_words_mean=round(sum(lengths) / max(len(lengths), 1), 2),
        image_size=args.size,
    )
    (OUT / "stats.json").write_text(json.dumps(dict(stats), indent=2))
    print(json.dumps(dict(stats), indent=2))


if __name__ == "__main__":
    main()
