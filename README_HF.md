---
title: Deepfake Detector
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Deepfake Detector (EfficientNet-B5)

A deepfake detection application using EfficientNet-B5 backbone.

## Features

- Image and video deepfake detection
- Real-time inference with probability visualization
- Configurable decision modes (threshold/argmax)
- Per-frame analysis for video inputs

## Usage

Upload an image or video to detect if it's real or AI-generated/manipulated.

## Model

This application uses a trained EfficientNet-B5 model. Ensure you have placed the trained model file at `models/efficientnet_b5_deepfake.h5`.
