from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from deepfake_detector.inference.predict import load_model, predict_image, predict_video


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
    if MODEL_CACHE["model"] is None:
        project_root = Path(__file__).resolve().parents[1]
        model_path = project_root / "models" / "efficientnet_b5_deepfake.h5"
        MODEL_CACHE["model"] = load_model(model_path)
    return MODEL_CACHE["model"]


def warmup_model():
    try:
        model = get_model()
        dummy = (np.zeros((224, 224, 3), dtype=np.uint8))
        _ = predict_image(model, dummy, image_size=MODEL_CACHE["image_size"])
    except Exception:
        pass


def _make_prob_plot(prob_real: float, prob_fake: float):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.bar(["Real", "Fake"], [prob_real, prob_fake], color=["#22c55e", "#ef4444"]) 
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def infer(image: np.ndarray, threshold: float = 0.5, decision_mode: str = "threshold", fast_mode: bool = False):
    """Run inference and decide label by selected decision mode.

    decision_mode:
      - 'threshold': label Real if P(real) >= threshold else Fake.
      - 'argmax': label of higher probability among {real,fake}.
    """
    if image is None:
        return {"Real": 0.0, "Fake": 0.0}, "Please upload an image."
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
            extra = f"Mode: argmax"
        else:  # threshold
            label = "Real" if prob_real >= float(threshold) else "Fake"
            confidence = prob_real if label == "Real" else prob_fake
            extra = f"Mode: threshold @ {threshold:.2f}"
        dt = time.perf_counter() - t0
        msg = (
            f"<div>Prediction: <span class='{'result-good' if label=='Real' else 'result-bad'}'>{label}</span> | Confidence: {confidence:.2%} | "
            f"P(real): {prob_real:.2%} | P(fake): {prob_fake:.2%} | {extra} | Size: {image_size[0]} | Time: {dt:.2f}s</div>"
        )
        plot = _make_prob_plot(prob_real, prob_fake)
        return {"Real": prob_real, "Fake": prob_fake}, msg, plot
    except Exception as e:
        return {"Real": 0.0, "Fake": 0.0}, f"Error: {e}", None


def build_ui():
    with gr.Blocks(title="Deepfake Detector (EfficientNet-B5)") as demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        gr.Markdown("# Deepfake Detector\nUse Image or Video input; choose decision mode and click Predict.")
        # Load suggested threshold if available
        suggested_threshold = 0.5
        try:
            import json
            project_root = Path(__file__).resolve().parents[1]
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

                def infer_video(path_or_array, frames_to_sample: int, threshold: float, decision_mode: str):
                    if path_or_array is None:
                        return {"Real": 0.0, "Fake": 0.0}, "Please upload a video."
                    try:
                        model = get_model()
                        try:
                            h, w = int(model.input_shape[1]), int(model.input_shape[2])
                            image_size = (h, w)
                        except Exception:
                            image_size = MODEL_CACHE["image_size"]
                        # Gradio may pass dict or str; ensure filepath string
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
                        msg = (
                            f"<div>Prediction: <span class='{'result-good' if label=='Real' else 'result-bad'}'>{label}</span> | Confidence: {confidence:.2%} | "
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

                btn_vid.click(infer_video, inputs=[vid, frames, threshold_vid, decision_mode_vid], outputs=[probs_vid, result_vid, frames_plot])
                clear_btn_vid.click(lambda: (None, {"Real": 0.0, "Fake": 0.0}, "", None), outputs=[vid, probs_vid, result_vid, frames_plot])

        # Examples removed per user request.

        # Disable warmup to avoid potential startup stalls
        # (can re-enable later once predictions are stable)
        # demo.load(lambda: warmup_model(), None, None)
    return demo


if __name__ == "__main__":
    app = build_ui()
    # Run without Gradio queue to avoid request stalls
    app.launch()
