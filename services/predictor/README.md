# Predictor Service

This service provides an API (FastAPI) for making predictions using a trained Naive Bayes model.

## Usage

- Loads trained models (pickle files) from a shared volume (`/models`).
- Provides endpoints to list available models and to make predictions with a selected model.

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Input
- Trained model file (pickle) in the `/models` directory.
- POST request with a record for prediction.

## Output
- Prediction result in JSON format. 