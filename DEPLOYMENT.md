# Deployment Guide

This guide covers different deployment options for the Deepfake Detector application.

## Prerequisites

Before deploying, ensure you have:
1. A trained model file (`efficientnet_b5_deepfake.h5`) in the `models/` directory
2. (Optional) A calibrated threshold file (`threshold.json`) in the `artifacts/` directory

## Deployment Options

### 1. Local Deployment

Run the application locally using Python:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The application will be available at `http://localhost:7860`.

### 2. Docker Deployment

Build and run using Docker:

```bash
# Build the Docker image
docker build -t deepfake-detector .

# Run the container
docker run -p 7860:7860 \
  -v $(pwd)/models:/app/deepfake_detector/models:ro \
  -v $(pwd)/artifacts:/app/deepfake_detector/artifacts:ro \
  deepfake-detector
```

Or using Docker Compose:

```bash
# Start the application
docker compose up

# Run in detached mode
docker compose up -d

# Stop the application
docker compose down
```

### 3. Hugging Face Spaces Deployment

To deploy on Hugging Face Spaces:

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select "Gradio" as the SDK
3. Upload the following files:
   - `app.py` (the root app.py file)
   - `requirements.txt`
   - `models/efficientnet_b5_deepfake.h5` (your trained model)
   - `artifacts/threshold.json` (optional)
4. The Space will automatically build and deploy

Alternatively, you can connect your GitHub repository to Hugging Face Spaces for automatic deployments.

### 4. Cloud Platform Deployment

#### Railway / Render / Fly.io

These platforms support Docker deployments. Use the provided `Dockerfile`:

1. Connect your GitHub repository
2. The platform will detect the `Dockerfile`
3. Set environment variables if needed:
   - `GRADIO_SERVER_NAME=0.0.0.0`
   - `GRADIO_SERVER_PORT=7860`

#### Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/deepfake-detector

# Deploy to Cloud Run
gcloud run deploy deepfake-detector \
  --image gcr.io/YOUR_PROJECT_ID/deepfake-detector \
  --port 7860 \
  --allow-unauthenticated
```

#### AWS Elastic Container Service (ECS)

1. Build and push the Docker image to Amazon ECR
2. Create an ECS task definition using the image
3. Deploy to ECS Fargate or EC2

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GRADIO_SERVER_NAME` | Server bind address | `127.0.0.1` (local) / `0.0.0.0` (Docker) |
| `GRADIO_SERVER_PORT` | Server port | `7860` |

## Model Files

The application expects the following files:

```
models/
  └── efficientnet_b5_deepfake.h5  # Required: Trained model

artifacts/
  └── threshold.json               # Optional: Calibrated threshold
```

### Model File Format

The model should be a Keras `.h5` file with:
- Input: Images of shape (batch, height, width, 3)
- Output: Sigmoid probability (P(real))

## Troubleshooting

### Model Not Found

Ensure the model file exists at the correct path. For Docker deployments, verify the volume mount is correct.

### Memory Issues

EfficientNet-B5 is memory-intensive. Ensure your deployment environment has at least:
- 2GB RAM for inference
- 4GB RAM recommended

### Port Already in Use

Change the port using environment variables:
```bash
GRADIO_SERVER_PORT=8080 python app.py
```

## Security Considerations

1. **Model Files**: Keep trained models secure; they may contain sensitive training data patterns
2. **Input Validation**: The application validates uploaded files
3. **Rate Limiting**: Consider adding rate limiting for production deployments
4. **HTTPS**: Use a reverse proxy (nginx, Traefik) for HTTPS in production

## Monitoring

For production deployments, consider adding:
- Health checks (Gradio provides a `/` endpoint)
- Logging aggregation
- Performance monitoring
