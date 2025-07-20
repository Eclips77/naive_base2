from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from naive_bayes.app import App
import uvicorn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
app = FastAPI()
data_app = App()

class FileRequest(BaseModel):
    file_name: str

class LabelRequest(BaseModel):
    target_column: str

class RecordRequest(BaseModel):
    record: dict

@app.get("/")
async def root():
    """Root endpoint to verify that the API is running.

    Usage:
        curl http://localhost:8000/
    """
    return {"message": "Welcome to the Naive Bayes Classifier API"}

@app.get("/available_files")
async def available_files():
    """Get a list of available CSV files.

    Returns:
        dict: List of file names.

    Usage:
        curl http://localhost:8000/available_files
    """
    files = data_app.list_available_files()
    if not files:
        raise HTTPException(status_code=404, detail="No CSV files found.")
    return {"files": files}

@app.post("/load_file")
async def load_file(req: FileRequest):
    """Load the selected file and return available columns.

    Args:
        req (FileRequest): Request containing file_name.
    Returns:
        dict: Status and list of columns.

    Usage:
        curl -X POST -H "Content-Type: application/json" \
            -d '{"file_name": "data.csv"}' http://localhost:8000/load_file
    """
    try:
        result = data_app.load_and_clean(req.file_name)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/set_target_column")
async def set_target_column(req: LabelRequest):
    """Set the target column for labels.

    Args:
        req (LabelRequest): Request containing target_column.

    Returns:
        dict: Status message.

    Usage:
        curl -X POST -H "Content-Type: application/json" \
            -d '{"target_column": "label"}' http://localhost:8000/set_target_column
    """
    try:
        result = data_app.set_target_column(req.target_column)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/features")
async def get_features():
    """Get feature columns (excluding target) with unique values.

    Returns:
        dict: Mapping from feature names to unique values.

    Usage:
        curl http://localhost:8000/features
    """
    try:
        return data_app.get_features_with_values()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train")
async def train_model():
    """Train the model using the selected target column.

    Returns:
        dict: Status message.

    Usage:
        curl -X POST http://localhost:8000/train
    """
    try:
        return data_app.train_model()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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

@app.post("/predict")
async def predict(req: RecordRequest):
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

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
