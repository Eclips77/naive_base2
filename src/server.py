from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from naive_bayes.app import App
import uvicorn
import os
app = FastAPI()
data_app = App()

# Automatically load data and train the model when the server starts
@app.on_event("startup")
async def startup_event():
    """Load data, set the target column and train the model on startup."""
    file_name = os.getenv("Data", "play_tennis.csv")
    try:
        # Load and clean the dataset
        data_app.load_and_clean(file_name)
        # Use the last column as the target by default
        target = data_app.train_df.columns[-1]
        data_app.set_target_column(target)
        # Train the model so the API is ready to serve predictions
        data_app.train_model()
    except Exception as e:
        # Log any issue during startup but allow the server to keep running
        print(f"Failed to initialise model: {e}")

class FileRequest(BaseModel):
    file_name: str

class LabelRequest(BaseModel):
    target_column: str

class RecordRequest(BaseModel):
    record: dict



@app.get("/accuracy")
async def get_accuracy():
    """Get the accuracy of the trained model.

    Returns:
        dict: Accuracy score.

    Usage:
        curl http://localhost:8000/accuracy
    """
    try:
        return data_app.evaluate_model()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/classify")
async def classify(req: RecordRequest):
    """Predict the class of a new record.

    Args:
        req (RecordRequest): Request containing record dictionary.

    Returns:
        dict: Prediction result.

    Usage:
        curl -X POST -H "Content-Type: application/json" \
            -d '{"record": {"feature": "value"}}' http://localhost:8000/predict
    """
    try:
        return data_app.classify_record(req.record)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000)
