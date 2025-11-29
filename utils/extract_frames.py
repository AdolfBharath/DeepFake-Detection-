import argparse
import os
from pathlib import Path

import cv2


def extract_first_frame(video_path: Path, out_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_path), frame)


def main():
    parser = argparse.ArgumentParser(description="Extract a first frame image from each video under src_root into dst_root/images")
    parser.add_argument("--src_root", required=True, help="Root containing class subfolders with videos: real/ and fake/")
    parser.add_argument("--dst_root", required=True, help="Destination root for images: will create real/ and fake/")
    parser.add_argument("--exts", default=".mp4,.avi,.mov,.mkv", help="Comma-separated video extensions to process")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    exts = set([e.strip().lower() for e in args.exts.split(",") if e.strip()])

    total = 0
    ok_count = 0
    for cls in ("real", "fake"):
        vdir = src_root / cls
        if not vdir.exists():
            continue
        for p in vdir.glob("**/*"):
            if p.is_file() and p.suffix.lower() in exts:
                total += 1
                out_name = p.stem + ".jpg"
                out_path = dst_root / cls / out_name
                if extract_first_frame(p, out_path):
                    ok_count += 1
    print(f"Processed {total} videos; wrote {ok_count} images to {dst_root}")


if __name__ == "__main__":
    main()
