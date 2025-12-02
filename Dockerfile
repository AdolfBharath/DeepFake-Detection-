# Deepfake Detector Docker Image
# Usage:
#   docker build -t deepfake-detector .
#   docker run -p 7860:7860 deepfake-detector
#
# For production, mount your trained model:
#   docker run -p 7860:7860 -v /path/to/models:/app/deepfake_detector/models:ro deepfake-detector

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code into deepfake_detector package
# This creates the package structure expected by the module imports:
#   /app/deepfake_detector/ui/app.py
#   /app/deepfake_detector/inference/predict.py
#   etc.
COPY . /app/deepfake_detector/

# Expose the port Gradio runs on
EXPOSE 7860

# Create models and artifacts directories (will be overlaid by volume mounts in production)
RUN mkdir -p /app/deepfake_detector/models /app/deepfake_detector/artifacts

# Run the Gradio app using the module structure
# The app will look for models in /app/deepfake_detector/models/
CMD ["python", "-m", "deepfake_detector.ui.app"]
