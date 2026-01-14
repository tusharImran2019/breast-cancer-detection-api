# Breast Cancer Detection API v2.0.0

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-green.svg)
![Flask](https://img.shields.io/badge/flask-2.3.3-orange.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.13.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

</div>

REST API for Breast Cancer Detection using Deep Learning. Classifies breast ultrasound images into Benign, Malignant, or Normal. Built with Flask, TensorFlow, and deployed on Render.

## 🚀 What's New in v2.0.0

- ✅ **Automatic Model Download**: Model automatically downloads from AWS S3 on first startup
- ✅ **Enhanced Error Handling**: Comprehensive error messages and validation
- ✅ **Structured Architecture**: Clean, modular code with separate classes for different concerns
- ✅ **Advanced Logging**: Detailed logging system for debugging and monitoring
- ✅ **Health Check Endpoint**: Monitor API and model status
- ✅ **Improved Security**: Better input validation and file size limits
- ✅ **CORS Enabled**: Ready for web and mobile app integration

## ✨ Features

- **Real-time Classification**: Instant breast cancer classification (Benign, Malignant, Normal)
- **3-Modality Fusion Model**: Advanced deep learning architecture
- **RESTful API**: Built with Flask for easy integration
- **Image Preprocessing**: Automatic image processing with OpenCV
- **Cloud Model Storage**: Model hosted on AWS S3 for seamless deployment
- **Comprehensive Validation**: File type, size, and format validation
- **Production Ready**: Deployed on Render with proper error handling

## 📋 API Endpoints

### 1. Health Check

**Endpoint:** `GET /`

**Description:** Returns API status and model information

**Response:**

```json
{
  "success": true,
  "service": "Breast Cancer Detection API",
  "version": "2.0.0",
  "model": "3-Modality Fusion (Image + Clinical + Genetic)",
  "status": "online",
  "model_loaded": true,
  "model_source": "AWS S3",
  "classes": ["Benign", "Malignant", "Normal"],
  "endpoints": {
    "GET /": "API status and information",
    "GET /health": "Health check endpoint",
    "POST /predict": "Image classification endpoint"
  },
  "author": "Md Tushar Imran",
  "contact": "tusharimran2019@gmail.com"
}
```

### 2. Model Health Check

**Endpoint:** `GET /health`

**Description:** Detailed health check for model status

**Response:**

```json
{
  "success": true,
  "status": "healthy",
  "model_loaded": true,
  "model_path": "breast_cancer_3modality_fusion.h5",
  "model_exists": true,
  "timestamp": "2025-01-26T10:30:00.123456"
}
```

### 3. Predict

**Endpoint:** `POST /predict`

**Description:** Classify breast ultrasound image

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Field: `image` (file upload)
- Supported formats: `png`, `jpg`, `jpeg`, `bmp`
- Max file size: `10 MB`

**Success Response:**

```json
{
  "success": true,
  "data": {
    "predicted_class": "Benign",
    "confidence": 95.67,
    "all_predictions": {
      "Benign": 95.67,
      "Malignant": 3.21,
      "Normal": 1.12
    },
    "timestamp": "2025-01-26T10:30:00.123456"
  },
  "message": "Detection complete: Benign (95.67% confidence)"
}
```

**Error Response:**

```json
{
  "success": false,
  "error": "No image provided. Please upload an image file."
}
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/tusharImran2019/breast-cancer-detection-api.git
cd breast-cancer-detection-api
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python main.py
```

The API will start on `http://0.0.0.0:5000`

**Note:** The model will be automatically downloaded from AWS S3 on first run. This may take a few minutes depending on your internet connection.

## 🧪 Testing the API

### Using cURL

```bash
# Health check
curl http://localhost:5000/

# Predict
curl -X POST -F "image=@path/to/your/ultrasound_image.jpg" http://localhost:5000/predict
```

### Using Python Requests

```python
import requests

# Health check
response = requests.get("http://localhost:5000/")
print(response.json())

# Predict
url = "http://localhost:5000/predict"
files = {"image": open("ultrasound_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Using Postman

1. Open Postman
2. Create a new `POST` request
3. URL: `http://localhost:5000/predict`
4. Body: Select `form-data`
5. Key: `image` (type: File)
6. Value: Upload your ultrasound image
7. Click **Send**

## 🚀 Deployment on Render

### Step 1: Create Web Service

1. Go to [Render](https://render.com/)
2. Click **New +** → **Web Service**
3. Connect your GitHub repository

### Step 2: Configure Service

- **Name**: breast-cancer-detection-api
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn main:app`
- **Instance Type**: Choose based on your needs

### Step 3: Deploy

Click **Create Web Service** and wait for deployment to complete.

### Step 4: Test Deployed API

```bash
curl https://your-app-name.onrender.com/
```

## 📊 Model Information

- **Model Name**: breast_cancer_3modality_fusion.h5
- **Model Source**: AWS S3 (https://uaecommerce.s3.ap-southeast-1.amazonaws.com/)
- **Input Size**: 128x128 pixels
- **Input Format**: RGB ultrasound images
- **Output Classes**: 3 classes (Benign, Malignant, Normal)
- **Architecture**: 3-Modality Fusion Deep Learning Model
- **Framework**: TensorFlow/Keras 2.13.0
- **Preprocessing**: Automatic resize and normalization

## 🏗️ Project Structure

```
breast-cancer-detection-api/
│
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── runtime.txt            # Python version for deployment
├── README.md              # Documentation
├── LICENSE                # MIT License
└── .gitignore            # Git ignore file
```

## 💻 Code Architecture

The application follows a clean, modular architecture:

- **Config Class**: Centralized configuration management
- **ModelManager Class**: Handles model downloading and loading
- **ImageProcessor Class**: Image validation and preprocessing
- **PredictionService Class**: Model prediction logic
- **Flask Routes**: RESTful API endpoints

## 📝 Dependencies

```
Flask==2.3.3
flask-cors==4.0.0
tensorflow==2.13.0
numpy==1.24.3
opencv-python-headless==4.8.0.76
gunicorn==21.2.0
gdown==4.7.1
requests
```

## 🔒 Security Features

- File type validation
- File size limits (10 MB max)
- Input sanitization
- Error handling and logging
- CORS configuration for controlled access

## 🐛 Troubleshooting

### Model Download Issues

If the model fails to download:

1. Check your internet connection
2. Verify the S3 URL is accessible
3. Check logs for specific error messages

### Memory Issues

If you encounter memory errors:

1. Ensure you have at least 2GB RAM available
2. Close other applications
3. Consider using a cloud deployment platform

### Import Errors

If you get import errors:

```bash
pip install -r requirements.txt --force-reinstall
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Md Tushar Imran**
- Email: tusharimran2019@gmail.com
- GitHub: [@tusharImran2019](https://github.com/tusharImran2019)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!

## 📚 Related Projects

- [Breast Cancer Classification Notebook](https://colab.research.google.com/drive/1ywmjcCCQ0-WQhov3QBOT-TsXA-Njk89w)
- [Project Report (Overleaf)](https://www.overleaf.com/project/69581255d91270d783d09a16)

---

<div align="center">
  Made with ❤️ by Md Tushar Imran
</div>
