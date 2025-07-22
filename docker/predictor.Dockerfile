FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY shared ./shared
COPY services/predictor ./predictor
COPY models ./models
EXPOSE 8000
ENTRYPOINT ["uvicorn", "predictor/main:app", "--host", "0.0.0.0", "--port", "8000"] 