import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def _run(cmd: List[str]):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def _kaggle_cmd(base_args: List[str]) -> List[str]:
    """Return a robust Kaggle CLI invocation for Windows.

    Prefer the 'kaggle' entrypoint if available; otherwise use
    the current Python interpreter with '-m kaggle' to avoid PATH issues.
    """
    exe = shutil.which("kaggle")
    if exe:
        return [exe] + base_args
    # Try common Windows path: <python_dir>\Scripts\kaggle.exe
    scripts_exe = Path(sys.executable).parent / "Scripts" / "kaggle.exe"
    if scripts_exe.exists():
        return [str(scripts_exe)] + base_args
    # Fallback to module (may fail if kaggle lacks __main__)
    return [sys.executable, "-m", "kaggle"] + base_args


def _extract_frames(video_path: Path, out_dir: Path, frames_per_video: int = 5, seed: int = 123):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: failed to open video {video_path}")
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        return 0
    # Pick evenly spaced frames
    indices = np.linspace(0, max(0, frame_count - 1), frames_per_video).astype(int)
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_file = out_dir / f"{video_path.stem}_f{idx}.jpg"
        cv2.imwrite(str(out_file), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1
    cap.release()
    return saved


def download_and_prepare_dfdc_preview(dataset_root: Path, downloads_dir: Path, frames_per_video: int = 5, val_ratio: float = 0.2):
    downloads_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = dataset_root.resolve()
    train_real = dataset_root / "train" / "real"
    train_fake = dataset_root / "train" / "fake"
    val_real = dataset_root / "val" / "real"
    val_fake = dataset_root / "val" / "fake"

    # Requires Kaggle credentials (~/.kaggle/kaggle.json). Accept competition rules first.
    zip_path = downloads_dir / "dfdc_preview_set.zip"
    if not zip_path.exists():
        try:
            _run(_kaggle_cmd(["competitions", "download", "-c", "deepfake-detection-challenge", "-f", "dfdc_preview_set.zip", "-p", str(downloads_dir)]))
        except FileNotFoundError:
            raise FileNotFoundError(
                "Kaggle CLI not found. Install with 'python -m pip install kaggle' and set up credentials in %USERPROFILE%/.kaggle/kaggle.json."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Kaggle download failed. Ensure kaggle.json is configured and competition rules accepted. Details: {e}"
            )

    extract_dir = downloads_dir / "dfdc_preview"
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    # Expect structure with metadata.json and a folder of videos
    meta_path = None
    for p in extract_dir.rglob("metadata.json"):
        meta_path = p
        break
    if meta_path is None:
        raise FileNotFoundError("Could not find metadata.json in the extracted preview.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta: Dict[str, Dict] = json.load(f)

    # Collect video files and labels
    videos: List[Tuple[Path, str]] = []
    for vid_name, info in meta.items():
        label = info.get("label", "")
        # Find actual path
        vid_path = None
        for ext in (".mp4", ".avi", ".mov"):
            cand = meta_path.parent / vid_name
            if cand.exists():
                vid_path = cand
                break
            cand = meta_path.parent / (Path(vid_name).stem + ext)
            if cand.exists():
                vid_path = cand
                break
        if vid_path is None:
            continue
        videos.append((vid_path, label))

    # Simple split: first N*(1-val) to train, rest to val (stratified-ish by alternating)
    real_videos = [v for v in videos if v[1].lower() == "real"]
    fake_videos = [v for v in videos if v[1].lower() == "fake"]

    def split(lst: List[Tuple[Path, str]]):
        n = int(len(lst) * (1 - val_ratio))
        return lst[:n], lst[n:]

    real_train, real_val = split(real_videos)
    fake_train, fake_val = split(fake_videos)

    counts = {"train": {"real": 0, "fake": 0}, "val": {"real": 0, "fake": 0}}
    for vid, _ in real_train:
        counts["train"]["real"] += _extract_frames(vid, train_real, frames_per_video)
    for vid, _ in fake_train:
        counts["train"]["fake"] += _extract_frames(vid, train_fake, frames_per_video)
    for vid, _ in real_val:
        counts["val"]["real"] += _extract_frames(vid, val_real, frames_per_video)
    for vid, _ in fake_val:
        counts["val"]["fake"] += _extract_frames(vid, val_fake, frames_per_video)

    print("Prepared DFDC preview frames at:", dataset_root)
    print("Counts:", counts)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "dataset"
    downloads_dir = project_root / "downloads"
    download_and_prepare_dfdc_preview(dataset_root, downloads_dir, frames_per_video=5, val_ratio=0.2)
