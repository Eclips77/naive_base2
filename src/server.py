import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from naive_bayes.app import App
import uvicorn

app = FastAPI()
data_app = App()

# Automatically load data and train the model when the server starts
@app.on_event("startup")
async def startup_event():
    """Load tennis data, clean it, set target column and train the model on startup."""
    file_name = "play_tennis.csv"
    try:
        print(f"Loading {file_name}...")
        # Load and clean the tennis dataset
        result = data_app.load_and_clean(file_name)
        print(f"Data loaded successfully with columns: {result['columns']}")
        
        # Use the last column (Play) as the target
        if data_app.train_df is not None:
            target = data_app.train_df.columns[-1]
            data_app.set_target_column(target)
            print(f"Target column set to: {target}")
        else:
            raise Exception("Failed to load training data")
        
        # Train the model
        train_result = data_app.train_model()
        print(f"Model training completed: {train_result['message']}")
        
        # Show model accuracy
        accuracy_result = data_app.evaluate_model()
        print(f"Model accuracy: {accuracy_result['accuracy']:.2%}")
        
        # Show available features for predictions
        features = data_app.get_features_with_values()
        print(f"Available features for prediction: {list(features.keys())}")
        print("API is ready to serve predictions!")
        
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        raise e

class RecordRequest(BaseModel):
    record: dict

@app.get("/")
async def root():
    """Root endpoint to verify that the API is running and show model info.

    Returns:
        dict: Welcome message and model status.

    Usage:
        curl http://localhost:8000/
    """
    try:
        # Get available features for user reference
        features = data_app.get_features_with_values()
        accuracy = data_app.evaluate_model()
        
        return {
            "message": "Welcome to the Tennis Naive Bayes Predictor API",
            "model_status": "Ready",
            "accuracy": f"{accuracy['accuracy']:.2%}",
            "available_features": features,
            "usage": "Send POST request to /predict with a record containing feature values"
        }
    except Exception as e:
        return {
            "message": "Welcome to the Tennis Naive Bayes Predictor API",
            "model_status": "Error",
            "error": str(e)
        }

@app.post("/predict")
async def predict(req: RecordRequest):
    """Predict whether tennis will be played based on weather conditions.

    Args:
        req (RecordRequest): Request containing record dictionary with weather features.

    Returns:
        dict: Prediction result.

    Example usage:
        curl -X POST -H "Content-Type: application/json" \
            -d '{"record": {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"}}' \
            http://localhost:8000/predict
    """
    try:
        result = data_app.classify_record(req.record)
        
        # Add confidence and feature validation
        features = data_app.get_features_with_values()
        
        # Validate that all required features are provided
        missing_features = []
        for feature in features.keys():
            if feature not in req.record:
                missing_features.append(feature)
        
        if missing_features:
            return {
                "error": f"Missing required features: {missing_features}",
                "required_features": features
            }
        
        # Validate feature values
        invalid_values = []
        for feature, value in req.record.items():
            if feature in features and value not in features[feature]:
                invalid_values.append(f"{feature}: '{value}' (valid: {features[feature]})")
        
        if invalid_values:
            return {
                "error": f"Invalid feature values: {invalid_values}",
                "provided_record": req.record,
                "valid_features": features
            }
        
        return {
            "prediction": result["prediction"],
            "input_record": req.record,
            "message": f"Prediction: {'Play tennis' if result['prediction'] == 'Yes' else 'Do not play tennis'}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


