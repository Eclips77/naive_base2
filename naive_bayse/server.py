from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Naive Bayes Classifier API"}

@app.get("/accuracy")
async def get_accuracy():
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

