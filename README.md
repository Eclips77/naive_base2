# Naive Bayes Microservices Project

This project implements a modular, SOLID-compliant Naive Bayes machine learning system using microservices architecture. Each service is fully decoupled and responsible for a single domain: training, prediction, or evaluation.

## Architecture

- **Trainer Service**: Loads, cleans, preprocesses, and trains a Naive Bayes model on a dataset. Saves the trained model to a shared volume.
- **Predictor Service**: Provides a FastAPI HTTP API for making predictions using any trained model in the shared volume.
- **Evaluator Service**: Loads a trained model and a test dataset, computes accuracy and evaluation metrics.
- **Shared Volume**: All models are saved and loaded from a shared Docker volume (`models-data`).

## Directory Structure

- `services/` - Contains the code for each microservice.
- `shared/` - Contains all shared logic (data loading, cleaning, model logic, etc.).
- `models/` - Directory for trained model files (shared via Docker volume).
- `docker/` - Dockerfiles for each service.
- `docker-compose.yml` - Orchestrates all services and the shared volume.

## Build & Run (Docker Compose)

1. **Build all services:**
   ```bash
   docker-compose build
   ```
2. **Run the services:**
   ```bash
   docker-compose up
   ```

## Usage

- **Trainer:**
  - Run interactively to train a model:
    ```bash
    docker-compose run trainer --data-file /app/path/to/data.csv --target-column TargetColumn --model-name my_model.pkl
    ```
- **Predictor:**
  - Access the API at [http://localhost:8000](http://localhost:8000)
  - List models:
    ```bash
    curl http://localhost:8000/models
    ```
  - Predict:
    ```bash
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"model_name": "my_model.pkl", "record": {"feature1": "value1", ...}}'
    ```
- **Evaluator:**
  - Run interactively to evaluate a model:
    ```bash
    docker-compose run evaluator --model-name my_model.pkl --test-file /app/path/to/test.csv --target-column TargetColumn
    ```

## Notes
- All code, documentation, and comments are in English only.
- Each service is fully modular and follows SOLID principles.
- You can extend or swap any component with minimal changes.

---

For more details, see the README in each service directory. 