import argparse
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def largest_face(faces: np.ndarray) -> Tuple[int, int, int, int]:
    # faces: [N, 4] in (x, y, w, h)
    if len(faces) == 0:
        return None
    areas = faces[:, 2] * faces[:, 3]
    idx = int(np.argmax(areas))
    return tuple(int(v) for v in faces[idx])


def crop_with_margin(x: int, y: int, w: int, h: int, margin: float, W: int, H: int):
    dx = int(w * margin)
    dy = int(h * margin)
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(W, x + w + dx)
    y1 = min(H, y + h + dy)
    return x0, y0, x1, y1


def process_image(src_path: Path, dst_path: Path, target_size: int = 224, margin: float = 0.2) -> bool:
    img = cv2.imread(str(src_path))
    if img is None:
        return False
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))

    if faces is None or len(faces) == 0:
        # Fallback: center crop square
        side = min(H, W)
        y0 = (H - side) // 2
        x0 = (W - side) // 2
        crop = img[y0:y0 + side, x0:x0 + side]
    else:
        x, y, w, h = largest_face(faces)
        x0, y0, x1, y1 = crop_with_margin(x, y, w, h, margin, W, H)
        crop = img[y0:y1, x0:x1]

    if crop.size == 0:
        return False
    crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    ensure_dir(dst_path.parent)
    cv2.imwrite(str(dst_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract face crops into a new dataset tree")
    parser.add_argument("--src_root", type=str, required=True, help="Source dataset root containing train/val/{real,fake}")
    parser.add_argument("--dst_root", type=str, required=True, help="Destination root for face-crop dataset")
    parser.add_argument("--size", type=int, default=224, help="Output image size (default: 224)")
    parser.add_argument("--margin", type=float, default=0.2, help="Face margin proportion (default: 0.2)")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    subsets = [("train", "real"), ("train", "fake"), ("val", "real"), ("val", "fake")]
    total_in, total_ok = 0, 0
    for split, cls in subsets:
        src_dir = src_root / split / cls
        dst_dir = dst_root / split / cls
        for img_path in list_images(src_dir):
            total_in += 1
            out_path = dst_dir / img_path.name
            ok = process_image(img_path, out_path, target_size=args.size, margin=args.margin)
            if ok:
                total_ok += 1

    print(f"Processed {total_in} images; successfully wrote {total_ok} face crops to {dst_root}")


if __name__ == "__main__":
    main()
