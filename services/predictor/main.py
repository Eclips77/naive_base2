import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from shared.predictor_manager import PredictorManager

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
manager = PredictorManager(models_dir=MODELS_DIR)

app = FastAPI(title="Naive Bayes Predictor Service")

class PredictionRequest(BaseModel):
    model_name: str
    record: Dict[str, Any]

class ModelListResponse(BaseModel):
    models: List[str]

class PredictionResponse(BaseModel):
    prediction: Any

@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"}

@app.get("/models", response_model=ModelListResponse, tags=["Models"])
def list_models():
    """
    List all available trained model files in the models directory.
    """
    return {"models": manager.list_models()}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    """
    Make a prediction using the specified model and input record.
    Args:
        request (PredictionRequest): Contains model_name and record (features).
    Returns:
        PredictionResponse: The prediction result.
    """
    try:
        prediction = manager.predict(request.model_name, request.record)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 