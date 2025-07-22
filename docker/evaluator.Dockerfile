FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY shared ./shared
COPY services/evaluator ./evaluator
COPY models ./models
ENTRYPOINT ["python", "evaluator/main.py"] 