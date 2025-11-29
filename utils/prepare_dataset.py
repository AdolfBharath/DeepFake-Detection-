import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split


ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}


def collect_images(folder: Path) -> List[Path]:
    files: List[Path] = []
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            files.append(p)
    return files


def copy_files(files: List[Path], dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = dest_dir / src.name
        # If name collision, append an index
        if dst.exists():
            stem, ext = dst.stem, dst.suffix
            i = 1
            while True:
                cand = dest_dir / f"{stem}_{i}{ext}"
                if not cand.exists():
                    dst = cand
                    break
                i += 1
        shutil.copy2(src, dst)


def prepare_dataset(real_src: Path, fake_src: Path, out_root: Path, val_ratio: float = 0.2, seed: int = 42):
    train_real = out_root / "train" / "real"
    train_fake = out_root / "train" / "fake"
    val_real = out_root / "val" / "real"
    val_fake = out_root / "val" / "fake"

    real_files = collect_images(real_src)
    fake_files = collect_images(fake_src)
    if not real_files or not fake_files:
        raise RuntimeError("No images found. Ensure both real and fake folders contain .jpg/.jpeg/.png files.")

    real_train, real_val = train_test_split(real_files, test_size=val_ratio, random_state=seed, shuffle=True)
    fake_train, fake_val = train_test_split(fake_files, test_size=val_ratio, random_state=seed, shuffle=True)

    # Copy
    copy_files(real_train, train_real)
    copy_files(fake_train, train_fake)
    copy_files(real_val, val_real)
    copy_files(fake_val, val_fake)

    print("Prepared dataset at:", out_root)
    print(f"Train: real={len(real_train)}, fake={len(fake_train)} | Val: real={len(real_val)}, fake={len(fake_val)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare folder-based dataset (train/val) from real/fake sources")
    parser.add_argument("--real_src", type=str, required=True, help="Path to folder containing REAL images")
    parser.add_argument("--fake_src", type=str, required=True, help="Path to folder containing FAKE images")
    parser.add_argument("--out_dir", type=str, default=None, help="Output dataset root (default: deepfake_detector/dataset)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio (default=0.2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_dir) if args.out_dir else project_root / "dataset"
    prepare_dataset(Path(args.real_src), Path(args.fake_src), out_root, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
