from typing import Dict, Tuple, Union, List
from pathlib import Path

import numpy as np
import tensorflow as tf
import cv2


def load_model(model_path: Union[str, Path]) -> tf.keras.Model:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Load without compiling to speed up load time in inference-only use
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def preprocess_image(img: np.ndarray, image_size: Tuple[int, int] = (224, 224)) -> tf.Tensor:
    """Preprocess a single RGB image for EfficientNet inference.

    Steps:
      - Ensure 3 channels.
      - Resize.
      - Convert to float32 and apply EfficientNet preprocess (expects 0-255 range internally).
      - Add batch dimension.
    Returns a Tensor suitable for model.predict.
    """
    if img.ndim == 2:  # grayscale -> RGB
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:  # RGBA -> RGB
        img = img[..., :3]
    img_tf = tf.convert_to_tensor(img, dtype=tf.float32)
    img_tf = tf.image.resize(img_tf, image_size)
    # EfficientNet preprocess_input expects pixels in [0,255]; currently in [0,255] if uint8, so keep scale
    if img_tf.dtype != tf.float32:
        img_tf = tf.cast(img_tf, tf.float32)
    # If image came in as uint8 converted to float32, values are still 0-255. If 0-1, scale up.
    max_val = tf.reduce_max(img_tf)
    img_tf = tf.cond(max_val <= 1.5, lambda: img_tf * 255.0, lambda: img_tf)
    img_tf = tf.keras.applications.efficientnet.preprocess_input(img_tf)
    img_tf = tf.expand_dims(img_tf, axis=0)
    return img_tf


def predict_image(
    model: tf.keras.Model,
    img: np.ndarray,
    image_size: Tuple[int, int] = (224, 224),
) -> Dict[str, float]:
    x = preprocess_image(img, image_size)
    # With tf.keras.utils.image_dataset_from_directory, class_names are
    # alphabetically sorted. In training we observed class_names=['fake','real'].
    # Binary labels are 0=fake, 1=real, so the model's sigmoid output corresponds
    # to P(real). Convert accordingly.
    prob_real = float(model.predict(x, verbose=0)[0][0])
    prob_fake = 1.0 - prob_real
    return {"real": prob_real, "fake": prob_fake}


def _sample_frame_indices(frame_count: int, num_frames: int) -> List[int]:
    if frame_count <= 0 or num_frames <= 0:
        return []
    if num_frames >= frame_count:
        return list(range(frame_count))
    step = frame_count / (num_frames + 1)
    return [max(0, min(frame_count - 1, int(round(step * (i + 1))))) for i in range(num_frames)]


def predict_video(
    model: tf.keras.Model,
    video_path: Union[str, Path],
    frames: int = 8,
    image_size: Tuple[int, int] = (224, 224),
) -> Dict[str, Union[float, int, List[float]]]:
    """Predict on multiple frames sampled from a video and aggregate.

    Returns a dict with average probabilities and per-frame real probabilities.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _sample_frame_indices(frame_count, frames)
    probs_real: List[float] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = predict_image(model, frame, image_size=image_size)
        probs_real.append(float(preds["real"]))
    cap.release()
    if not probs_real:
        return {"real": 0.0, "fake": 0.0, "frames_used": 0, "per_frame_real": []}
    avg_real = float(np.mean(probs_real))
    avg_fake = 1.0 - avg_real
    return {"real": avg_real, "fake": avg_fake, "frames_used": len(probs_real), "per_frame_real": probs_real}


if __name__ == "__main__":
    # Minimal dry-run with random image (sanity check)
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "efficientnet_b5_deepfake.h5"
    if model_path.exists():
        model = load_model(model_path)
        dummy = (np.random.rand(256, 256, 3) * 255).astype("uint8")
        preds = predict_image(model, dummy, image_size=(224, 224))
        print("Prediction (random image):", preds)
    else:
        print(f"Model not found at {model_path}. Train the model first.")
