"""Download MS-COCO 2017 *val* split (5,000 images, ~25k captions).

val2017 is the right size for this milestone (1k-5k pairs) and avoids the
18 GB train download. Files land in data/raw/.

    python scripts/download_coco.py
"""
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = {
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",  # ~241 MB
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",  # ~778 MB
}


class _Bar(tqdm):
    def hook(self, blocks=1, bsize=1, total=None):
        if total:
            self.total = total
        self.update(blocks * bsize - self.n)


def fetch(name: str, url: str) -> Path:
    dest = RAW / name
    if dest.exists():
        print(f"[skip] {name} already downloaded")
        return dest
    with _Bar(unit="B", unit_scale=True, desc=name) as bar:
        urllib.request.urlretrieve(url, dest, reporthook=bar.hook)
    return dest


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        z = fetch(name, url)
        marker = RAW / (z.stem + ".extracted")
        if not marker.exists():
            print(f"[unzip] {name}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(RAW)
            marker.touch()
    print(f"Done -> {RAW}")


if __name__ == "__main__":
    main()
