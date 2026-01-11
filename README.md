# breast-cancer-detection-api
REST API for Breast Cancer Detection using Deep Learning. Classifies breast ultrasound images into Benign, Malignant, or Normal. Built with Flask, TensorFlow, and deployed on Render.

## Features
- Real-time breast cancer classification (Benign, Malignant, Normal)
- RESTful API built with Flask
- Deep learning model using TensorFlow
- Image preprocessing with OpenCV
- CORS enabled for mobile app integration
- Easy deployment on Render

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/tusharImran2019/breast-cancer-detection-api.git
cd breast-cancer-detection-api
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Model File
Upload your trained model file `breast_cancer_3modality_fusion.h5` to the root directory of the project.

### 4. Run Locally
```bash
python app.py
```
The API will start on `http://0.0.0.0:5000`

## API Endpoints

### Health Check
```bash
GET /
```
Returns API status and model information.

### Predict
```bash
POST /predict
```
**Request:** Send image file as multipart/form-data with key `image`

**Response:**
```json
{
  "success": true,
  "prediction": "Benign",
  "confidence": 95.67,
  "all_predictions": {
    "Benign": 95.67,
    "Malignant": 3.21,
    "Normal": 1.12
  },
  "message": "Detection: Benign (95.67% confidence)"
}
```

## Testing the API

### Using cURL
```bash
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/predict
```

### Using Python
```python
import requests

url = "http://localhost:5000/predict"
files = {"image": open("test_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
4. Upload the model file to Render (or use external storage)
5. Deploy!

## Model Information
- **Input:** Breast ultrasound images (128x128 pixels)
- **Output:** Classification into Benign, Malignant, or Normal
- **Architecture:** 3-Modality Fusion Deep Learning Model
- **Framework:** TensorFlow/Keras

## License
MIT License
