import argparse
from pathlib import Path
import cv2
import math
import os


def sample_frame_indices(frame_count: int, num_frames: int) -> list:
    """Return a list of frame indices approximately evenly spaced.
    Skips first/last frame margins slightly to avoid black/transition frames.
    """
    if frame_count <= 0 or num_frames <= 0:
        return []
    if num_frames >= frame_count:
        return list(range(frame_count))
    # Even spacing: use fractional positions then round
    step = frame_count / (num_frames + 1)
    return [max(0, min(frame_count - 1, int(round(step * (i + 1))))) for i in range(num_frames)]


def detect_and_crop_face(frame, margin: float = 0.2):
    """Detect largest face and crop with margin; fallback to center square crop."""
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
    if faces is None or len(faces) == 0:
        side = min(H, W)
        y0 = (H - side) // 2
        x0 = (W - side) // 2
        return frame[y0:y0 + side, x0:x0 + side]
    # largest face
    areas = faces[:, 2] * faces[:, 3]
    idx = int(areas.argmax())
    x, y, w, h = faces[idx]
    dx = int(w * margin)
    dy = int(h * margin)
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(W, x + w + dx)
    y1 = min(H, y + h + dy)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return frame
    return crop


def process_video(video_path: Path, out_dir: Path, num_frames: int, face_crop: bool, target_size: int, margin: float) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(frame_count, num_frames)
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if face_crop:
            frame = detect_and_crop_face(frame, margin=margin)
        # Resize square preserving aspect by padding if needed
        h, w = frame.shape[:2]
        if h != w:
            side = max(h, w)
            pad_color = [0, 0, 0]
            padded = cv2.copyMakeBorder(frame,
                                        top=(side - h) // 2,
                                        bottom=(side - h) - (side - h) // 2,
                                        left=(side - w) // 2,
                                        right=(side - w) - (side - w) // 2,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=pad_color)
            frame = padded
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{video_path.stem}_f{idx}.jpg"
        out_path = out_dir / out_name
        if cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            saved += 1
    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract multiple evenly spaced frames per video into image dataset.")
    parser.add_argument("--src_root", required=True, help="Root containing class subfolders with videos: real/ and fake/")
    parser.add_argument("--dst_root", required=True, help="Destination root for images (creates real/ and fake/ subdirs)")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to sample per video")
    parser.add_argument("--face_crop", action="store_true", help="Enable face detection + margin cropping")
    parser.add_argument("--margin", type=float, default=0.2, help="Face margin proportion if --face_crop")
    parser.add_argument("--size", type=int, default=224, help="Output image size (square)")
    parser.add_argument("--exts", default=".mp4,.avi,.mov,.mkv", help="Comma-separated video extensions")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    exts = {e.strip().lower() for e in args.exts.split(',') if e.strip()}

    total_videos = 0
    total_images = 0
    for cls in ("real", "fake"):
        vdir = src_root / cls
        if not vdir.exists():
            continue
        for p in vdir.glob("**/*"):
            if p.is_file() and p.suffix.lower() in exts:
                total_videos += 1
                out_dir = dst_root / cls
                saved = process_video(p, out_dir, args.frames, args.face_crop, args.size, args.margin)
                total_images += saved
    print(f"Processed {total_videos} videos; wrote {total_images} images to {dst_root} (≈{args.frames} per video)")


if __name__ == "__main__":
    main()
