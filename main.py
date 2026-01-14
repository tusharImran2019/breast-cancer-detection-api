"""")
Breast Cancer Detection API - 3-Modality Fusion Model
Built with Flask, TensorFlow, OpenCV
Model: breast_cancer_3modality_fusion.h5 (from AWS S3)
Author: Md Tushar Imran
Version: 2.0.0
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
import requests
from functools import wraps
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================
class Config:
    """Application configuration"""
    MODEL_URL = 'https://uaecommerce.s3.ap-southeast-1.amazonaws.com/breast_cancer_3modality_fusion.h5'
    MODEL_PATH = 'breast_cancer_3modality_fusion.h5'
    IMAGE_SIZE = (128, 128)
    CLASS_LABELS = ['Benign', 'Malignant', 'Normal']
    MAX_IMAGE_SIZE_MB = 10
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# ============================================
# Model Management
# ============================================
class ModelManager:
    """Manages model downloading and loading"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
    
    def download_model(self):
        """Download model from S3 if not present"""
        try:
            if os.path.exists(Config.MODEL_PATH):
                logger.info(f"Model already exists at {Config.MODEL_PATH}")
                return True
            
            logger.info(f"Downloading model from S3: {Config.MODEL_URL}")
            response = requests.get(Config.MODEL_URL, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(Config.MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}%")
            
            logger.info("Model downloaded successfully!")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading model: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            return False
    
    def load_model(self):
        """Load the TensorFlow model"""
        try:
            if not os.path.exists(Config.MODEL_PATH):
                logger.error(f"Model file not found: {Config.MODEL_PATH}")
                return False
            
            logger.info("Loading model...")
            self.model = tf.keras.models.load_model(Config.MODEL_PATH)
            self.model_loaded = True
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.model_loaded = False
            return False
    
    def initialize(self):
        """Initialize model (download + load)"""
        logger.info("Initializing model...")
        if self.download_model():
            return self.load_model()
        return False

# ============================================
# Image Processing
# ============================================
class ImageProcessor:
    """Handles image preprocessing"""
    
    @staticmethod
    def validate_image(file):
        """Validate uploaded image file"""
        if not file:
            return False, "No file provided"
        
        # Check file extension
        filename = file.filename.lower()
        if not any(filename.endswith(ext) for ext in Config.ALLOWED_EXTENSIONS):
            return False, f"Invalid file type. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        
        # Check file size
        file.seek(0, os.SEEK_END)
        size_mb = file.tell() / (1024 * 1024)
        file.seek(0)
        
        if size_mb > Config.MAX_IMAGE_SIZE_MB:
            return False, f"File too large. Max size: {Config.MAX_IMAGE_SIZE_MB}MB"
        
        return True, "Valid"
    
    @staticmethod
    def preprocess(image_bytes):
        """Preprocess image for model prediction"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image")
            
            # Resize to model input size
            img = cv2.resize(img, Config.IMAGE_SIZE)
            
            # Normalize pixel values to [0, 1]
            img = img.astype('float32') / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img, None
            
        except Exception as e:
            return None, f"Preprocessing error: {str(e)}"

# ============================================
# Prediction Service
# ============================================
class PredictionService:
    """Handles model predictions"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def predict(self, processed_image):
        """Make prediction on preprocessed image"""
        try:
            if not self.model_manager.model_loaded:
                return None, "Model not loaded"
            
            # Get predictions
            predictions = self.model_manager.model.predict(processed_image, verbose=0)
            
            # Extract results
            predicted_idx = np.argmax(predictions[0])
            predicted_class = Config.CLASS_LABELS[predicted_idx]
            confidence = float(predictions[0][predicted_idx]) * 100
            
            # All class probabilities
            all_predictions = {
                label: round(float(predictions[0][i]) * 100, 2)
                for i, label in enumerate(Config.CLASS_LABELS)
            }
            
            result = {
                'predicted_class': predicted_class,
                'confidence': round(confidence, 2),
                'all_predictions': all_predictions,
                'timestamp': datetime.now().isoformat()
            }
            
            return result, None
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, f"Prediction failed: {str(e)}"

# ============================================
# Flask Application
# ============================================

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize services
model_manager = ModelManager()
prediction_service = PredictionService(model_manager)

# Initialize model on startup
logger.info("Starting Breast Cancer Detection API...")
if not model_manager.initialize():
    logger.warning("Model initialization failed. API running in degraded mode.")

# ============================================
# Decorators
# ============================================
def require_model(f):
    """Decorator to check if model is loaded"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not model_manager.model_loaded:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please try again later.'
            }), 503
        return f(*args, **kwargs)
    return decorated_function

# ============================================
# Routes
# ============================================

@app.route('/', methods=['GET'])
def home():
    """API status endpoint"""
    return jsonify({
        'success': True,
        'service': 'Breast Cancer Detection API',
        'version': '2.0.0',
        'model': '3-Modality Fusion (Image + Clinical + Genetic)',
        'status': 'online',
        'model_loaded': model_manager.model_loaded,
        'model_source': 'AWS S3',
        'classes': Config.CLASS_LABELS,
        'endpoints': {
            'GET /': 'API status and information',
            'GET /health': 'Health check endpoint',
            'POST /predict': 'Image classification endpoint'
        },
        'author': 'Md Tushar Imran',
        'contact': 'tusharimran2019@gmail.com'
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    health_status = {
        'success': True,
        'status': 'healthy' if model_manager.model_loaded else 'degraded',
        'model_loaded': model_manager.model_loaded,
        'model_path': Config.MODEL_PATH,
        'model_exists': os.path.exists(Config.MODEL_PATH),
        'timestamp': datetime.now().isoformat()
    }
    
    status_code = 200 if model_manager.model_loaded else 503
    return jsonify(health_status), status_code

@app.route('/predict', methods=['POST'])
@require_model
def predict():
    """Image classification endpoint"""
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided. Please upload an image file.'
            }), 400
        
        image_file = request.files['image']
        
        # Validate image
        is_valid, message = ImageProcessor.validate_image(image_file)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
        # Read image bytes
        image_bytes = image_file.read()
        
        # Preprocess image
        processed_image, error = ImageProcessor.preprocess(image_bytes)
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        # Make prediction
        result, error = prediction_service.predict(processed_image)
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 500
        
        # Return successful response
        return jsonify({
            'success': True,
            'data': result,
            'message': f"Detection complete: {result['predicted_class']} ({result['confidence']}% confidence)"
        }), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error occurred'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'The requested URL was not found on this server.'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred. Please try again later.'
    }), 500

# ============================================
# Main Entry Point
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
