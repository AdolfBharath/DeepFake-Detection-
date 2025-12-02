# Deepfake Detector Docker Image
# Usage:
#   docker build -t deepfake-detector .
#   docker run -p 7860:7860 deepfake-detector

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code into deepfake_detector package
COPY . /app/deepfake_detector/

# Expose the port Gradio runs on
EXPOSE 7860

# Create models and artifacts directories
RUN mkdir -p /app/deepfake_detector/models /app/deepfake_detector/artifacts

# Run the Gradio app
CMD ["python", "-m", "deepfake_detector.ui.app"]
