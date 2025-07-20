# Naive Bayes Classifier API

This project exposes a simple Naive Bayes classifier via FastAPI. Models are automatically saved after training to `model.pkl` in the project root. You can reload a saved model using the `/load_model` endpoint.

## Usage

1. **Train and Save**
   ```bash
   curl -X POST http://localhost:8000/train
   ```
   Training stores the classifier in `model.pkl`.

2. **Load a Saved Model**
   ```bash
   curl -X POST -H "Content-Type: application/json" \
        -d '{"path": "model.pkl"}' http://localhost:8000/load_model
   ```

The saved file `model.pkl` resides in the repository's base directory.
