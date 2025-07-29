# Tennis Naive Bayes Predictor

A containerized application that predicts whether tennis should be played based on weather conditions using a Naive Bayes classifier.

## 🚀 Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed on your system

### Running the Application

1. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```

2. **Or build and run manually:**
   ```bash
   # Build the image
   docker build -t tennis-predictor .
   
   # Run the container
   docker run -p 8000:8000 tennis-predictor
   ```

3. **Access the API:**
   - API will be available at: `http://localhost:8000`
   - API documentation: `http://localhost:8000/docs`

### Testing the API

**Check server status:**
```bash
curl http://localhost:8000/
```

**Make a prediction:**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"record": {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak"}}' \
  http://localhost:8000/predict
```

## 🖥️ Running the Streamlit Client

The Streamlit client provides a user-friendly interface to interact with the prediction API.

1. **Install requirements for client:**
   ```bash
   pip install streamlit requests
   ```

2. **Run the client:**
   ```bash
   streamlit run ui/client.py
   ```

3. **Access the client:**
   - Open your browser to: `http://localhost:8501`

## 📊 Features

- **Automatic Model Training**: The model trains automatically when the container starts
- **Weather-based Predictions**: Predict tennis playability based on:
  - Outlook (Sunny, Overcast, Rain)
  - Temperature (Hot, Mild, Cool)
  - Humidity (High, Normal)
  - Wind (Strong, Weak)
- **Input Validation**: Ensures all required features are provided with valid values
- **Health Checks**: Container includes health monitoring
- **RESTful API**: Simple endpoints for predictions

## 🛠️ Development

### Project Structure
```
naive_base2/
├── Data/
│   └── play_tennis.csv      # Training dataset
├── naive_bayes/
│   ├── app.py              # Main application logic
│   ├── data_loader.py      # Data loading utilities
│   └── data_processor.py   # Data preprocessing
├── src/
│   ├── evaluator.py        # Model evaluation
│   ├── naive_bayes_classifier.py  # Classifier implementation
│   └── server.py           # FastAPI server
├── ui/
│   └── client.py           # Streamlit client interface
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Docker Compose setup
└── requirements.txt        # Python dependencies
```

### API Endpoints

- `GET /` - Server status and model information
- `POST /predict` - Make predictions with weather data

### Container Management

**Stop the container:**
```bash
docker-compose down
```

**View logs:**
```bash
docker-compose logs tennis-predictor
```

**Rebuild after changes:**
```bash
docker-compose up --build
```

## 📈 Model Information

- **Algorithm**: Naive Bayes Classifier
- **Dataset**: Tennis decision dataset based on weather conditions
- **Training**: Automatic on container startup
- **Accuracy**: Displayed in API status and client interface

## 🔧 Configuration

- **Port**: 8000 (configurable in docker-compose.yml)
- **Dataset**: `Data/play_tennis.csv` (mounted as read-only volume)
- **Health Check**: Built-in container health monitoring
