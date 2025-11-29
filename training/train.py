import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import tensorflow as tf

from deepfake_detector.models.efficientnet_b5 import build_efficientnet_b5
from deepfake_detector.utils.preprocessing import get_datasets


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def plot_history(history: tf.keras.callbacks.History, out_dir: Path):
    ensure_dir(out_dir)
    hist = history.history

    # Accuracy
    plt.figure()
    plt.plot(hist.get("accuracy", []), label="train_acc")
    plt.plot(hist.get("val_accuracy", []), label="val_acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_accuracy.png")
    plt.close()

    # Loss
    plt.figure()
    plt.plot(hist.get("loss", []), label="train_loss")
    plt.plot(hist.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B5 Deepfake Detector")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fine_tune", type=str, default="all", choices=["all", "top", "none"])
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--model_out", type=str, default=None, help="Path to save best model")
    parser.add_argument("--monitor", type=str, default="val_accuracy", choices=["val_accuracy", "val_loss"])
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    artifacts_dir = project_root / "artifacts"
    ensure_dir(models_dir)
    ensure_dir(artifacts_dir)

    image_size = (args.image_size, args.image_size)

    print(f"Loading datasets from: {args.data_dir}")
    train_ds, val_ds, test_ds, class_names = get_datasets(
        args.data_dir, image_size=image_size, batch_size=args.batch_size, augment=args.augment
    )
    print(f"Classes: {class_names}")

    print("Building model EfficientNet-B5 ...")
    model = build_efficientnet_b5(
        input_shape=(args.image_size, args.image_size, 3),
        learning_rate=args.learning_rate,
        fine_tune=args.fine_tune,
    )
    model.summary()

    best_model_path = (
        Path(args.model_out)
        if args.model_out is not None
        else models_dir / "efficientnet_b5_deepfake.h5"
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=args.monitor, patience=7, restore_best_weights=True, mode="max" if args.monitor == "val_accuracy" else "min"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor=args.monitor,
            save_best_only=True,
            save_weights_only=False,
            mode="max" if args.monitor == "val_accuracy" else "min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=args.monitor,
            factor=0.5,
            patience=3,
            mode="max" if args.monitor == "val_accuracy" else "min",
            verbose=1,
        ),
    ]

    print("Starting training ...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print(f"Saving training curves to: {artifacts_dir}")
    plot_history(history, artifacts_dir)

    # Evaluate on test if exists, otherwise on val
    eval_name = "test"
    eval_ds = test_ds
    if eval_ds is None:
        eval_name = "validation"
        eval_ds = val_ds

    print(f"Evaluating on {eval_name} set ...")
    loss, acc = model.evaluate(eval_ds)
    metrics = {f"{eval_name}_loss": float(loss), f"{eval_name}_accuracy": float(acc)}
    with open(artifacts_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Final metrics:", metrics)

    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
