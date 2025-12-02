"""
Hugging Face Spaces entry point for Deepfake Detector.

This file serves as the main entry point for Hugging Face Spaces deployment.
It provides the same functionality as the main app but with imports adjusted
for the Spaces environment.

NOTE: This file intentionally duplicates code from inference/predict.py and ui/app.py
to create a standalone, self-contained entry point that works on Hugging Face Spaces
without requiring the full module structure. This is a common pattern for Spaces
deployments where the app.py file must be self-contained.
"""

import os
import sys
from pathlib import Path

# Add the current directory to path for proper imports when running standalone
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from typing import Dict, Tuple, Union, List

import gradio as gr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


# ============ Model Loading and Inference (from inference/predict.py) ============

def load_model(model_path: Union[str, Path]) -> tf.keras.Model:
    """Load a trained Keras model from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def preprocess_image(img: np.ndarray, image_size: Tuple[int, int] = (224, 224)) -> tf.Tensor:
    """Preprocess a single RGB image for EfficientNet inference."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img_tf = tf.convert_to_tensor(img, dtype=tf.float32)
    img_tf = tf.image.resize(img_tf, image_size)
    if img_tf.dtype != tf.float32:
        img_tf = tf.cast(img_tf, tf.float32)
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
    """Run prediction on a single image."""
    x = preprocess_image(img, image_size)
    prob_real = float(model.predict(x, verbose=0)[0][0])
    prob_fake = 1.0 - prob_real
    return {"real": prob_real, "fake": prob_fake}


def _sample_frame_indices(frame_count: int, num_frames: int) -> List[int]:
    """Sample evenly spaced frame indices from a video."""
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
    """Predict on multiple frames sampled from a video and aggregate."""
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


# ============ Gradio UI (from ui/app.py) ============

MODEL_CACHE = {"model": None, "image_size": (224, 224)}

CUSTOM_CSS = """
.gradio-container {background: linear-gradient(135deg, #0b1021, #1b1f3b 60%, #111827); color: #e5e7eb;}
.gradio-container .tabs {background: transparent;}
.gradio-container .tabitem {background: rgba(255,255,255,0.04); border-radius: 10px;}
.gradio-container .markdown {color: #e5e7eb;}
button {background: #4f46e5 !important; color: #ffffff !important; border: none !important; box-shadow: 0 6px 16px rgba(79,70,229,0.35);} 
button:hover {filter: brightness(1.06);} 
.result-good {color: #22c55e; font-weight: 600;}
.result-bad {color: #ef4444; font-weight: 600;}
"""


def get_model() -> tf.keras.Model:
    """Load and cache the model."""
    if MODEL_CACHE["model"] is None:
        project_root = Path(__file__).resolve().parent
        model_path = project_root / "models" / "efficientnet_b5_deepfake.h5"
        MODEL_CACHE["model"] = load_model(model_path)
    return MODEL_CACHE["model"]


def _make_prob_plot(prob_real: float, prob_fake: float):
    """Create a bar chart of probabilities."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.bar(["Real", "Fake"], [prob_real, prob_fake], color=["#22c55e", "#ef4444"]) 
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def infer(image: np.ndarray, threshold: float = 0.5, decision_mode: str = "threshold", fast_mode: bool = False):
    """Run inference on an image."""
    if image is None:
        return {"Real": 0.0, "Fake": 0.0}, "Please upload an image.", None
    try:
        import time
        t0 = time.perf_counter()
        model = get_model()
        try:
            h, w = int(model.input_shape[1]), int(model.input_shape[2])
            image_size = (h, w)
        except Exception:
            image_size = MODEL_CACHE["image_size"]
        preds = predict_image(model, image, image_size=image_size)
        prob_real = preds["real"]
        prob_fake = preds["fake"]

        if decision_mode == "argmax":
            label = "Real" if prob_real >= prob_fake else "Fake"
            confidence = max(prob_real, prob_fake)
            extra = "Mode: argmax"
        else:
            label = "Real" if prob_real >= float(threshold) else "Fake"
            confidence = prob_real if label == "Real" else prob_fake
            extra = f"Mode: threshold @ {threshold:.2f}"
        dt = time.perf_counter() - t0
        css_class = 'result-good' if label == 'Real' else 'result-bad'
        msg = (
            f"<div>Prediction: <span class='{css_class}'>{label}</span> | Confidence: {confidence:.2%} | "
            f"P(real): {prob_real:.2%} | P(fake): {prob_fake:.2%} | {extra} | Size: {image_size[0]} | Time: {dt:.2f}s</div>"
        )
        plot = _make_prob_plot(prob_real, prob_fake)
        return {"Real": prob_real, "Fake": prob_fake}, msg, plot
    except Exception as e:
        return {"Real": 0.0, "Fake": 0.0}, f"Error: {e}", None


def infer_video(path_or_array, frames_to_sample: int, threshold: float, decision_mode: str):
    """Run inference on a video."""
    if path_or_array is None:
        return {"Real": 0.0, "Fake": 0.0}, "Please upload a video.", None
    try:
        model = get_model()
        try:
            h, w = int(model.input_shape[1]), int(model.input_shape[2])
            image_size = (h, w)
        except Exception:
            image_size = MODEL_CACHE["image_size"]
        if isinstance(path_or_array, dict) and "name" in path_or_array:
            video_path = path_or_array["name"]
        else:
            video_path = path_or_array
        out = predict_video(model, video_path, frames=int(frames_to_sample), image_size=image_size)
        prob_real = float(out["real"]) if "real" in out else 0.0
        prob_fake = float(out["fake"]) if "fake" in out else 0.0
        frames_used = int(out.get("frames_used", 0))
        if decision_mode == "argmax":
            label = "Real" if prob_real >= prob_fake else "Fake"
            confidence = max(prob_real, prob_fake)
            extra = "Mode: argmax"
        else:
            label = "Real" if prob_real >= float(threshold) else "Fake"
            confidence = prob_real if label == "Real" else prob_fake
            extra = f"Mode: threshold @ {threshold:.2f}"
        css_class = 'result-good' if label == 'Real' else 'result-bad'
        msg = (
            f"<div>Prediction: <span class='{css_class}'>{label}</span> | Confidence: {confidence:.2%} | "
            f"Avg P(real): {prob_real:.2%} | Avg P(fake): {prob_fake:.2%} | Frames used: {frames_used} | {extra}</div>"
        )
        per = out.get("per_frame_real", [])
        fig, ax = plt.subplots(figsize=(4, 2.5))
        if per:
            ax.plot(range(1, len(per)+1), per, marker="o", color="#4f46e5")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Frame #")
        ax.set_ylabel("P(real)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return {"Real": prob_real, "Fake": prob_fake}, msg, fig
    except Exception as e:
        return {"Real": 0.0, "Fake": 0.0}, f"Error: {e}", None


def build_ui():
    """Build and return the Gradio UI."""
    with gr.Blocks(title="Deepfake Detector (EfficientNet-B5)") as demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        gr.Markdown("# Deepfake Detector\nUse Image or Video input; choose decision mode and click Predict.")
        
        # Load suggested threshold if available
        suggested_threshold = 0.5
        try:
            import json
            project_root = Path(__file__).resolve().parent
            th_path = project_root / "artifacts" / "threshold.json"
            if th_path.exists():
                with open(th_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    val_raw = data.get("best_threshold", 0.5)
                    try:
                        val = float(val_raw) if val_raw is not None else 0.5
                    except Exception:
                        val = 0.5
                    if not (0.0 <= val <= 1.0):
                        val = 0.5
                    suggested_threshold = val
        except Exception:
            suggested_threshold = 0.5

        with gr.Tabs():
            with gr.TabItem("Image"):
                with gr.Row():
                    with gr.Column():
                        img = gr.Image(type="numpy", label="Input Image")
                        threshold_img = gr.Slider(0.0, 1.0, value=float(suggested_threshold), step=0.01, label="Threshold (Real if P(real) ≥)")
                        decision_mode_img = gr.Radio(choices=["threshold", "argmax"], value="threshold", label="Decision Mode")
                        fast_mode_img = gr.Checkbox(label="Fast Mode (experimental)", value=False)
                        with gr.Row():
                            btn_img = gr.Button("Predict", variant="primary")
                            clear_btn_img = gr.Button("Clear")
                    with gr.Column():
                        probs_img = gr.Label(num_top_classes=2, label="Probabilities (P(real), P(fake))")
                        result_img = gr.HTML(label="Prediction Details")
                        plot_img = gr.Plot(label="Probability Chart")

                btn_img.click(infer, inputs=[img, threshold_img, decision_mode_img, fast_mode_img], outputs=[probs_img, result_img, plot_img])
                clear_btn_img.click(lambda: (None, {"Real": 0.0, "Fake": 0.0}, "", None), outputs=[img, probs_img, result_img, plot_img])

            with gr.TabItem("Video"):
                with gr.Row():
                    with gr.Column():
                        vid = gr.Video(label="Input Video", interactive=True)
                        frames = gr.Slider(1, 32, value=8, step=1, label="Frames to Sample")
                        threshold_vid = gr.Slider(0.0, 1.0, value=float(suggested_threshold), step=0.01, label="Threshold (Real if P(real) ≥)")
                        decision_mode_vid = gr.Radio(choices=["threshold", "argmax"], value="threshold", label="Decision Mode")
                        with gr.Row():
                            btn_vid = gr.Button("Predict Video", variant="primary")
                            clear_btn_vid = gr.Button("Clear")
                    with gr.Column():
                        probs_vid = gr.Label(num_top_classes=2, label="Average Probabilities (P(real), P(fake))")
                        result_vid = gr.HTML(label="Prediction Details")
                        frames_plot = gr.Plot(label="Per-frame P(real)")

                btn_vid.click(infer_video, inputs=[vid, frames, threshold_vid, decision_mode_vid], outputs=[probs_vid, result_vid, frames_plot])
                clear_btn_vid.click(lambda: (None, {"Real": 0.0, "Fake": 0.0}, "", None), outputs=[vid, probs_vid, result_vid, frames_plot])

    return demo


# Create and launch the app
demo = build_ui()

if __name__ == "__main__":
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port)
