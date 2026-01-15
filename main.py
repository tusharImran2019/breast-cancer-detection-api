""""
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
import sys
import requests
from functools import wraps
import logging
from datetime import datetime
import time
import traceback

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
    MODEL_URL = 'https://bacbonschool.s3.ap-south-1.amazonaws.com/uploads/e-Teachers_Guide/breast_ultrasound_classifier.h5'
    MODEL_PATH = 'breast_ultrasound_classifier.h5'
    IMAGE_SIZE = (128, 128)
    CLASS_LABELS = ['Benign', 'Malignant', 'Normal']
    MAX_IMAGE_SIZE_MB = 10
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    MODEL_LOAD_RETRIES = 3
    MODEL_LOAD_RETRY_DELAY = 5  # seconds

# ============================================
# Model Management
# ============================================
class ModelManager:
    """Manages model downloading and loading"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.load_error = None
        self.model_file_size = 0
    
    def download_model(self):
        """Download model from S3 if not present"""
        try:
            if os.path.exists(Config.MODEL_PATH):
                file_size = os.path.getsize(Config.MODEL_PATH)
                self.model_file_size = file_size
                size_mb = file_size / (1024 * 1024)
                logger.info(f"Model already exists at {Config.MODEL_PATH} ({size_mb:.2f} MB)")
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
                            if downloaded % (1024 * 1024) == 0:  # Log every MB
                                logger.info(f"Download progress: {progress:.1f}% ({downloaded / (1024*1024):.1f} MB)")
            
            self.model_file_size = os.path.getsize(Config.MODEL_PATH)
            size_mb = self.model_file_size / (1024 * 1024)
            logger.info(f"Model downloaded successfully! Size: {size_mb:.2f} MB")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading model: {e}")
            logger.error(f"Request exception details: {type(e).__name__}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _load_model_with_compatibility_fix(self):
        """Load model with compatibility fix for batch_shape issue"""
        import h5py
        import json
        import tempfile
        import shutil
        
        try:
            # Create a temporary copy to modify
            temp_model_path = Config.MODEL_PATH + '.tmp'
            shutil.copy2(Config.MODEL_PATH, temp_model_path)
            
            # Read the model file and fix the config
            with h5py.File(temp_model_path, 'r+') as f:
                # Navigate to model config
                if 'model_config' in f.attrs:
                    model_config_str = f.attrs['model_config']
                    if isinstance(model_config_str, bytes):
                        model_config_str = model_config_str.decode('utf-8')
                    model_config = json.loads(model_config_str)
                    
                    # Fix layer configs recursively
                    def fix_dtype_policy(obj):
                        """Recursively fix DTypePolicy objects"""
                        if isinstance(obj, dict):
                            # Check if this is a DTypePolicy object
                            if obj.get('class_name') == 'DTypePolicy':
                                # Convert to string dtype
                                dtype_name = obj.get('config', {}).get('name', 'float32')
                                return dtype_name
                            
                            # Recursively process all values
                            fixed_obj = {}
                            for key, value in obj.items():
                                fixed_obj[key] = fix_dtype_policy(value)
                            return fixed_obj
                        elif isinstance(obj, list):
                            return [fix_dtype_policy(item) for item in obj]
                        else:
                            return obj
                    
                    def fix_layer_config(layer_config):
                        if isinstance(layer_config, dict):
                            # Fix InputLayer batch_shape
                            if layer_config.get('class_name') == 'InputLayer':
                                if 'config' in layer_config:
                                    config = layer_config['config']
                                    if 'batch_shape' in config:
                                        batch_shape = config['batch_shape']
                                        if batch_shape and len(batch_shape) > 1:
                                            config['input_shape'] = batch_shape[1:]
                                        del config['batch_shape']
                            
                            # Fix DTypePolicy recursively in config
                            if 'config' in layer_config:
                                layer_config['config'] = fix_dtype_policy(layer_config['config'])
                            
                            # Recursively fix nested layers
                            for key, value in layer_config.items():
                                if isinstance(value, (dict, list)):
                                    fix_layer_config(value)
                        elif isinstance(layer_config, list):
                            for item in layer_config:
                                fix_layer_config(item)
                    
                    # First fix DTypePolicy in the entire config
                    model_config = fix_dtype_policy(model_config)
                    # Then fix layer-specific issues
                    fix_layer_config(model_config)
                    
                    # Write back the fixed config
                    f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
            
            # Now load the model with fixed config
            model = tf.keras.models.load_model(temp_model_path, compile=False)
            
            # Clean up temp file
            try:
                os.remove(temp_model_path)
            except:
                pass
            
            return model
            
        except Exception as e:
            logger.warning(f"Compatibility fix method failed: {e}")
            # Fallback: try loading with custom objects
            return self._load_model_with_custom_objects()
    
    def _load_model_with_custom_objects(self):
        """Load model with custom objects for compatibility"""
        # Create custom objects to handle compatibility issues
        custom_objects = {}
        
        # Custom InputLayer to handle batch_shape
        class CompatibleInputLayer(tf.keras.layers.InputLayer):
            @classmethod
            def from_config(cls, config):
                if 'batch_shape' in config:
                    batch_shape = config.pop('batch_shape')
                    if batch_shape and len(batch_shape) > 1:
                        config['input_shape'] = tuple(batch_shape[1:])
                return super().from_config(config)
        custom_objects['InputLayer'] = CompatibleInputLayer
        
        # Handle DTypePolicy - convert to string dtype
        class DTypePolicyCompat:
            def __init__(self, name='float32'):
                self.name = name
            @classmethod
            def from_config(cls, config):
                return cls(config.get('name', 'float32'))
        
        # Try loading with custom objects
        try:
            return tf.keras.models.load_model(
                Config.MODEL_PATH,
                compile=False,
                custom_objects=custom_objects
            )
        except Exception as e:
            logger.warning(f"Loading with custom objects failed: {e}")
            # Last resort: try with compile=False and safe_mode=False
            try:
                return tf.keras.models.load_model(
                    Config.MODEL_PATH,
                    compile=False,
                    safe_mode=False
                )
            except:
                raise e
    
    def load_model(self):
        """Load the TensorFlow model with retry mechanism"""
        if not os.path.exists(Config.MODEL_PATH):
            error_msg = f"Model file not found: {Config.MODEL_PATH}"
            logger.error(error_msg)
            self.load_error = error_msg
            return False
        
        # Check file size
        file_size = os.path.getsize(Config.MODEL_PATH)
        size_mb = file_size / (1024 * 1024)
        logger.info(f"Model file found. Size: {size_mb:.2f} MB")
        
        if file_size == 0:
            error_msg = "Model file is empty (0 bytes)"
            logger.error(error_msg)
            self.load_error = error_msg
            return False
        
        # Try loading with retries
        for attempt in range(1, Config.MODEL_LOAD_RETRIES + 1):
            try:
                logger.info(f"Loading model... (Attempt {attempt}/{Config.MODEL_LOAD_RETRIES})")
                
                # Check TensorFlow version and model compatibility
                logger.info(f"TensorFlow version: {tf.__version__}")
                try:
                    keras_version = tf.keras.__version__
                    logger.info(f"Keras version: {keras_version}")
                except AttributeError:
                    # Keras version not available in this TF version
                    try:
                        import keras
                        logger.info(f"Keras version: {keras.__version__}")
                    except:
                        logger.info("Keras version: (integrated with TensorFlow)")
                
                # Try loading with different strategies for compatibility
                load_attempts = [
                    # Strategy 1: Standard load
                    lambda: tf.keras.models.load_model(Config.MODEL_PATH, compile=False),
                    # Strategy 2: With safe_mode=False
                    lambda: tf.keras.models.load_model(Config.MODEL_PATH, compile=False, safe_mode=False),
                    # Strategy 3: Handle batch_shape compatibility issue
                    lambda: self._load_model_with_compatibility_fix(),
                ]
                
                last_error = None
                for i, load_func in enumerate(load_attempts, 1):
                    try:
                        logger.info(f"Trying load strategy {i}...")
                        self.model = load_func()
                        logger.info(f"Model loaded successfully with strategy {i}")
                        break
                    except Exception as e:
                        logger.warning(f"Load strategy {i} failed: {str(e)}")
                        last_error = e
                        if i == len(load_attempts):
                            raise last_error
                
                # Verify model structure
                if self.model is None:
                    raise ValueError("Model loaded but is None")
                
                # Test model with dummy input
                try:
                    test_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
                    _ = self.model.predict(test_input, verbose=0)
                    logger.info("Model structure verified successfully")
                except Exception as test_error:
                    logger.warning(f"Model structure test failed: {test_error}")
                    # Don't fail completely, model might still work
                
                self.model_loaded = True
                self.load_error = None
                logger.info("Model loaded successfully!")
                
                # Log model summary
                try:
                    logger.info(f"Model input shape: {self.model.input_shape}")
                    logger.info(f"Model output shape: {self.model.output_shape}")
                except:
                    pass
                
                return True
                
            except MemoryError as e:
                error_msg = f"Memory error loading model (Attempt {attempt}): {str(e)}"
                logger.error(error_msg)
                self.load_error = error_msg
                if attempt < Config.MODEL_LOAD_RETRIES:
                    logger.info(f"Retrying in {Config.MODEL_LOAD_RETRY_DELAY} seconds...")
                    time.sleep(Config.MODEL_LOAD_RETRY_DELAY)
                else:
                    logger.error("All retry attempts failed due to memory error")
                    return False
                    
            except tf.errors.NotFoundError as e:
                error_msg = f"Model file corrupted or incompatible (Attempt {attempt}): {str(e)}"
                logger.error(error_msg)
                self.load_error = error_msg
                logger.error(f"Full error: {traceback.format_exc()}")
                return False  # Don't retry for corrupted files
                
            except Exception as e:
                error_msg = f"Error loading model (Attempt {attempt}): {str(e)}"
                logger.error(error_msg)
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                self.load_error = error_msg
                
                if attempt < Config.MODEL_LOAD_RETRIES:
                    logger.info(f"Retrying in {Config.MODEL_LOAD_RETRY_DELAY} seconds...")
                    time.sleep(Config.MODEL_LOAD_RETRY_DELAY)
                else:
                    logger.error("All retry attempts failed")
                    self.model = None
                    self.model_loaded = False
                    return False
        
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
        self._model_input_info = None
    
    def _get_model_input_info(self):
        """Get information about model inputs"""
        if self._model_input_info is not None:
            return self._model_input_info
        
        model = self.model_manager.model
        if model is None:
            return None
        
        # Check if model has multiple inputs
        try:
            if hasattr(model, 'inputs') and model.inputs:
                inputs = model.inputs
                self._model_input_info = {
                    'num_inputs': len(inputs),
                    'input_shapes': [inp.shape.as_list() if hasattr(inp.shape, 'as_list') else list(inp.shape) for inp in inputs],
                    'input_names': [inp.name for inp in inputs]
                }
                logger.info(f"Model has {len(inputs)} inputs: {self._model_input_info}")
            else:
                self._model_input_info = {
                    'num_inputs': 1,
                    'input_shapes': [model.input_shape if hasattr(model, 'input_shape') else None],
                    'input_names': ['input']
                }
        except Exception as e:
            logger.warning(f"Could not get model input info: {e}")
            self._model_input_info = {'num_inputs': 1, 'input_shapes': [None], 'input_names': ['input']}
        
        return self._model_input_info
    
    def _prepare_multi_input(self, processed_image):
        """Prepare inputs for multi-input model (3-Modality Fusion)"""
        input_info = self._get_model_input_info()
        
        if input_info is None or input_info['num_inputs'] == 1:
            return processed_image
        
        # Model expects multiple inputs (Image, Clinical, Genetic)
        num_inputs = input_info['num_inputs']
        input_shapes = input_info['input_shapes']
        
        logger.info(f"Preparing {num_inputs} inputs for multi-modality model")
        
        inputs = []
        for i, shape in enumerate(input_shapes):
            if shape is None:
                inputs.append(processed_image)
                continue
            
            # Check if this is the image input (typically has 4 dimensions: batch, height, width, channels)
            if len(shape) == 4 and shape[-1] == 3:
                # This is the image input
                inputs.append(processed_image)
                logger.info(f"Input {i}: Image input with shape {shape}")
            else:
                # Create dummy input for clinical/genetic data
                # Shape is typically [None, num_features] or [batch, num_features]
                dummy_shape = [1] + [s if s is not None else 1 for s in shape[1:]]
                dummy_input = np.zeros(dummy_shape, dtype=np.float32)
                inputs.append(dummy_input)
                logger.info(f"Input {i}: Dummy input created with shape {dummy_shape}")
        
        return inputs
    
    def predict(self, processed_image):
        """Make prediction on preprocessed image"""
        try:
            if not self.model_manager.model_loaded:
                return None, "Model not loaded"
            
            # Prepare inputs (handles multi-input models)
            model_inputs = self._prepare_multi_input(processed_image)
            
            # Get predictions
            predictions = self.model_manager.model.predict(model_inputs, verbose=0)
            
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
            logger.error(f"Traceback: {traceback.format_exc()}")
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
logger.info(f"Python version: {sys.version}")
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"NumPy version: {np.__version__}")
logger.info(f"OpenCV version: {cv2.__version__}")

if not model_manager.initialize():
    logger.warning("Model initialization failed. API running in degraded mode.")
    if model_manager.load_error:
        logger.error(f"Load error details: {model_manager.load_error}")

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
            'POST /predict': 'Image classification endpoint',
            'POST /reload-model': 'Manually reload model (debugging)'
        },
        'author': 'Md Tushar Imran',
        'contact': 'tusharimran2019@gmail.com'
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_exists = os.path.exists(Config.MODEL_PATH)
    model_size = 0
    if model_exists:
        model_size = os.path.getsize(Config.MODEL_PATH)
    
    health_status = {
        'success': True,
        'status': 'healthy' if model_manager.model_loaded else 'degraded',
        'model_loaded': model_manager.model_loaded,
        'model_path': Config.MODEL_PATH,
        'model_exists': model_exists,
        'model_size_mb': round(model_size / (1024 * 1024), 2) if model_exists else 0,
        'load_error': model_manager.load_error if not model_manager.model_loaded else None,
        'tensorflow_version': tf.__version__,
        'timestamp': datetime.now().isoformat()
    }
    
    status_code = 200 if model_manager.model_loaded else 503
    return jsonify(health_status), status_code

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """Manually reload the model (useful for debugging)"""
    try:
        logger.info("Manual model reload requested")
        success = model_manager.load_model()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model reloaded successfully',
                'model_loaded': model_manager.model_loaded,
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to reload model',
                'load_error': model_manager.load_error,
                'model_loaded': model_manager.model_loaded,
                'timestamp': datetime.now().isoformat()
            }), 500
            
    except Exception as e:
        logger.error(f"Error in manual reload: {e}")
        return jsonify({
            'success': False,
            'error': f'Reload failed: {str(e)}'
        }), 500

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
