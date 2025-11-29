import argparse
import os
import random
import string
import sys
from pathlib import Path

import urllib.request


def _rand_name(prefix: str = "img", ext: str = "jpg") -> str:
    token = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{prefix}_{token}.{ext}"


def download_picsum_portraits(dst_dir: Path, count: int = 100, size: int = 800) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Picsum photos provide random images; while not guaranteed faces, it's royalty-free and simple.
    # For better quality faces, replace this with Unsplash API (requires API key).
    downloaded = 0
    for _ in range(count):
        fname = dst_dir / _rand_name("real")
        url = f"https://picsum.photos/{size}/{size}"
        try:
            urllib.request.urlretrieve(url, str(fname))
            downloaded += 1
        except Exception:
            continue
    return downloaded


def make_fake_placeholders(src_dir: Path, dst_dir: Path, max_count: int = 100) -> int:
    """Bootstrap 'fake' by copying some real images as placeholders.
    NOTE: These are NOT true deepfakes; replace with actual deepfake samples later.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpeg"))
    random.shuffle(files)
    copied = 0
    for p in files[:max_count]:
        try:
            target = dst_dir / _rand_name("fake", p.suffix.replace(".", "") or "jpg")
            target.write_bytes(p.read_bytes())
            copied += 1
        except Exception:
            continue
    return copied


def main():
    parser = argparse.ArgumentParser(description="Download royalty-free portraits and bootstrap dataset")
    parser.add_argument("--raw_root", required=False, default=str(Path(__file__).resolve().parents[1] / "raw"), help="Root raw folder containing real/ and fake/")
    parser.add_argument("--real_count", type=int, default=200, help="Number of real portraits to download")
    parser.add_argument("--fake_count", type=int, default=200, help="Number of fake placeholders to create")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    real_dir = raw_root / "real"
    fake_dir = raw_root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.real_count} portraits to: {real_dir}")
    n_real = download_picsum_portraits(real_dir, count=args.real_count)
    print(f"Downloaded real portraits: {n_real}")

    print(f"Bootstrapping {args.fake_count} fake placeholders to: {fake_dir}")
    n_fake = make_fake_placeholders(real_dir, fake_dir, max_count=args.fake_count)
    print(f"Created fake placeholders: {n_fake}")

    print("Done. Next steps:\n - Run extract_faces.py to crop faces\n - Run prepare_dataset.py to split train/val")


if __name__ == "__main__":
    sys.exit(main())
