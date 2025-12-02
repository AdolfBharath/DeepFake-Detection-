# Deepfake Detector (EfficientNet‑B5)

![CI](https://github.com/AdolfBharath/DeepFake-Detection-/actions/workflows/python-ci.yml/badge.svg)

A complete image/video deepfake detector using EfficientNet‑B5. It classifies inputs as Real or Fake, with training, evaluation, and a polished Gradio UI.

**Highlights**
- EfficientNet‑B5 backbone with transfer learning
- Image + Video inference (frame sampling and aggregation)
- Decision modes: threshold or argmax
- Calibrated threshold and ROC/AUC evaluation
- Modern UI with styled results and probability charts

A complete image-based deepfake detector using EfficientNet-B5. It classifies images as Real or Fake and includes training, inference, and a local Gradio UI.

## Project Structure

```
deepfake_detector/
│── dataset/ (you provide this)
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/ (optional)
│       ├── real/
│       └── fake/
│── models/
│   └── (trained models saved here)
│── artifacts/
│   └── (training curves & metrics)
│── models/
│   └── efficientnet_b5.py
│── training/
│   └── train.py
│── inference/
│   └── predict.py
│── ui/
│   └── app.py
│── utils/
│   └── preprocessing.py
│── requirements.txt
│── README.md
```

## Setup

1. Create and activate a Python environment (recommended).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r deepfake_detector\requirements.txt
```

3. Prepare your dataset in the folder structure shown above. Example root: `deepfake_detector\dataset`.

## Training

Run the training script (uses EfficientNet-B5 pretrained on ImageNet, binary cross-entropy, Adam, EarlyStopping, and ModelCheckpoint):

```powershell
python -m deepfake_detector.training.train --data_dir deepfake_detector\dataset --epochs 15 --batch_size 16 --image_size 224 --augment --monitor val_accuracy
```

- Best model is saved to: `deepfake_detector\models\efficientnet_b5_deepfake.h5`
- Training curves saved to: `deepfake_detector\artifacts\training_accuracy.png` and `training_loss.png`
- Final metrics (validation if no test set) saved to: `deepfake_detector\artifacts\final_metrics.json`

Optional flags:
- `--fine_tune` one of `all|top|none` (default: `all`)
- `--learning_rate` (default: 1e-4)
- `--model_out` custom save path

## Inference (Script)

After training, run a quick test with a single image in Python:

```python
import numpy as np
from PIL import Image
from pathlib import Path
from deepfake_detector.inference.predict import load_model, predict_image

model = load_model(Path('deepfake_detector/models/efficientnet_b5_deepfake.h5'))
img = np.array(Image.open('path_to_image.jpg').convert('RGB'))
preds = predict_image(model, img, image_size=(224, 224))
print('Probabilities:', preds)  # {'real': ..., 'fake': ...}
```

## UI (Gradio)

Launch the local Gradio app for interactive inference:

```powershell
python -m deepfake_detector.ui.app
```

Upload an image and click Predict. The app shows the predicted label (Real/Fake) and confidence.

### Decision Modes

You can choose between two decision strategies in the UI:
- `threshold`: Classifies as Real if `P(real) ≥ threshold`, otherwise Fake. The threshold is loaded from `artifacts/threshold.json` when available.
- `argmax`: Ignores threshold and simply picks the class with the higher probability.

The model outputs a sigmoid value interpreted as `P(real)`. The fake probability reported is `P(fake) = 1 - P(real)`.

### Multi-Frame Extraction (Videos → Images)

To build a richer image dataset from video sources, use the multi-frame extractor:

```powershell
python -m deepfake_detector.utils.extract_multi_frames --src_root raw\\videos --dst_root dataset_frames --frames 8 --face_crop --size 224
```

Arguments:
- `--src_root`: Root folder containing `real/` and `fake/` subfolders with video files.
- `--dst_root`: Destination root where extracted frames are stored under `real/` and `fake/`.
- `--frames`: Number of evenly spaced frames sampled per video (default 5).
- `--face_crop`: Enable largest-face detection (Haar) with margin fallback.
- `--margin`: Margin around detected face (default 0.2).
- `--size`: Output square image size (default 224).

After extraction, you can split into train/val using your existing dataset preparation script or manually arrange into the expected directory structure.

### Video Inference
## Project Structure

```
deepfake_detector/
├── artifacts/              # metrics, curves, threshold
├── evaluation/             # ROC/AUC, threshold calibration
├── inference/              # predict.py (image & video)
├── models/                 # saved checkpoints
├── training/               # training script
├── ui/                     # Gradio app
├── utils/                  # dataset utilities
└── requirements.txt
```

## CI

GitHub Actions runs a Python CI on push/PR: installs dependencies, compiles sources for syntax, and performs smoke imports.

## Notes

You can upload a video directly in the UI (Video tab). The app samples N evenly spaced frames, runs the image model per frame, and aggregates probabilities (mean). The final label is chosen by your decision mode:

- `threshold`: Real if average `P(real) ≥ threshold` else Fake.
- `argmax`: Picks the higher of average `P(real)` vs `P(fake)`.

CLI-style usage via Python snippet:

```python
from deepfake_detector.inference.predict import load_model, predict_video
model = load_model('deepfake_detector/models/efficientnet_b5_deepfake.h5')
out = predict_video(model, 'path_to_video.mp4', frames=8, image_size=(224, 224))
print(out)  # {'real': 0.53, 'fake': 0.47, 'frames_used': 8, 'per_frame_real': [...]} 
```

## Notes

- Hardware: EfficientNet-B5 is large; consider smaller batch size if memory-constrained.
- Data: Ensure sufficient examples in both classes to avoid imbalance issues.
- Augmentation: Light augmentations (flip, rotate, zoom, contrast, brightness jitter) are applied to training data.
- Preprocessing: Images are resized to the chosen `--image_size` and normalized with EfficientNet preprocessing.

## Deployment

The application can be deployed using several methods:

### Quick Start (Local)

```bash
pip install -r requirements.txt
python app.py
```

### Docker

```bash
docker build -t deepfake-detector .
docker run -p 7860:7860 -v $(pwd)/models:/app/deepfake_detector/models:ro deepfake-detector
```

### Hugging Face Spaces

Upload `app.py`, `requirements.txt`, and your trained model to a new Gradio Space.

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## License

For educational and research purposes. Provide your own dataset and ensure you have rights to use it.