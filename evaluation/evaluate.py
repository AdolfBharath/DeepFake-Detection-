import json
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, confusion_matrix

from deepfake_detector.inference.predict import load_model
from deepfake_detector.utils.preprocessing import get_datasets


def compute_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, dict]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Maximize Youden's J statistic (tpr - fpr)
    j_scores = tpr - fpr
    idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[idx])
    return best_thresh, {
        "auc": float(auc(fpr, tpr)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "best_index": int(idx),
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "efficientnet_b5_deepfake.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(model_path)

    # Load validation dataset only
    data_dir = project_root / "dataset"
    _, val_ds, _, class_names = get_datasets(str(data_dir), image_size=(model.input_shape[1], model.input_shape[2]), batch_size=32, augment=False)
    if class_names != ["fake", "real"]:
        print("Warning: Expected class order ['fake','real'], got:", class_names)

    y_true = []
    y_prob = []
    for batch_x, batch_y in val_ds:
        probs = model.predict(batch_x, verbose=0).reshape(-1)
        y_prob.extend(probs.tolist())
        y_true.extend(batch_y.numpy().reshape(-1).tolist())

    y_true = np.array(y_true, dtype=np.float32)
    y_prob = np.array(y_prob, dtype=np.float32)

    best_thresh, roc_info = compute_best_threshold(y_true, y_prob)

    y_pred = (y_prob >= best_thresh).astype(np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = float((tp + tn) / (tp + tn + fp + fn))

    out = {
        "best_threshold": best_thresh,
        "accuracy": acc,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "roc": roc_info,
    }
    with open(artifacts_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Saved:", artifacts_dir / "threshold.json")
    print("Suggested threshold:", best_thresh)
    print("Validation accuracy @ threshold:", acc)


if __name__ == "__main__":
    main()
