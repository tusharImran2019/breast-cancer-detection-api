"""
Local Image Prediction Script
Uses breast_cancer_3modality_fusion.h5 model from AWS S3
Author: Md Tushar Imran
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import requests
import h5py
import json
import shutil

# ============================================
# Configuration
# ============================================
MODEL_URL = 'https://uaecommerce.s3.ap-southeast-1.amazonaws.com/breast_cancer_3modality_fusion.h5'
MODEL_PATH = 'breast_cancer_3modality_fusion.h5'
IMAGE_SIZE = (128, 128)
CLASS_LABELS = ['Benign', 'Malignant', 'Normal']

# Global model variable for caching
_cached_model = None


def download_model():
    """Download model from S3 if not present"""
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✓ Model already exists ({size_mb:.2f} MB)")
        return True
    
    print(f"⬇ Downloading model from S3...")
    print(f"  URL: {MODEL_URL}")
    
    try:
        response = requests.get(MODEL_URL, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r  Progress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✓ Model downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False


def fix_model_config():
    """Fix model config for TensorFlow compatibility"""
    print("🔧 Fixing model config for compatibility...")
    
    temp_model_path = MODEL_PATH + '.tmp'
    shutil.copy2(MODEL_PATH, temp_model_path)
    
    try:
        with h5py.File(temp_model_path, 'r+') as f:
            if 'model_config' in f.attrs:
                model_config_str = f.attrs['model_config']
                if isinstance(model_config_str, bytes):
                    model_config_str = model_config_str.decode('utf-8')
                model_config = json.loads(model_config_str)
                
                def fix_config_recursively(obj, parent_key=None):
                    """Recursively fix all compatibility issues"""
                    if isinstance(obj, dict):
                        # If this is a DTypePolicy, convert it
                        if obj.get('class_name') == 'DTypePolicy':
                            return None  # Remove dtype policy entirely, TF will use defaults
                        
                        # Fix InputLayer batch_shape
                        if obj.get('class_name') == 'InputLayer' and 'config' in obj:
                            config = obj['config']
                            if 'batch_shape' in config:
                                batch_shape = config.pop('batch_shape')
                                if batch_shape and len(batch_shape) > 1:
                                    config['input_shape'] = list(batch_shape[1:])
                            # Remove dtype if it's complex
                            if 'dtype' in config and isinstance(config['dtype'], dict):
                                del config['dtype']
                        
                        # Fix other layer configs
                        if 'config' in obj and isinstance(obj['config'], dict):
                            config = obj['config']
                            # Remove complex dtype fields
                            if 'dtype' in config and isinstance(config['dtype'], dict):
                                del config['dtype']
                        
                        # Recursively process
                        result = {}
                        for k, v in obj.items():
                            fixed_v = fix_config_recursively(v, k)
                            if fixed_v is not None or k != 'dtype':
                                result[k] = fixed_v if fixed_v is not None else v
                        return result
                    elif isinstance(obj, list):
                        return [fix_config_recursively(item) for item in obj]
                    return obj
                
                model_config = fix_config_recursively(model_config)
                f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
        
        print("✓ Model config fixed")
        return temp_model_path
        
    except Exception as e:
        print(f"⚠ Config fix failed: {e}")
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        return None


def load_model():
    """Load the model (cached for future predictions)"""
    global _cached_model
    
    if _cached_model is not None:
        print("✓ Using cached model")
        return _cached_model
    
    if not os.path.exists(MODEL_PATH):
        if not download_model():
            return None
    
    print(f"🔄 Loading model...")
    print(f"  TensorFlow version: {tf.__version__}")
    
    # Apply monkey patches for compatibility BEFORE loading
    _apply_compatibility_patches()
    
    # Try different loading strategies
    strategies = [
        ("Standard load", lambda p: tf.keras.models.load_model(p, compile=False)),
        ("Safe mode disabled", lambda p: tf.keras.models.load_model(p, compile=False, safe_mode=False)),
    ]
    
    for name, load_func in strategies:
        try:
            print(f"  Trying: {name}...")
            _cached_model = load_func(MODEL_PATH)
            print(f"✓ Model loaded with: {name}")
            return _cached_model
        except Exception as e:
            print(f"  ✗ {name} failed: {str(e)[:100]}")
    
    # Try with compatibility fix
    print("  Trying: Compatibility fix (HDF5 config modification)...")
    temp_path = fix_model_config()
    if temp_path:
        try:
            _cached_model = tf.keras.models.load_model(temp_path, compile=False)
            print("✓ Model loaded with compatibility fix")
            os.remove(temp_path)
            return _cached_model
        except Exception as e:
            print(f"  ✗ Compatibility fix failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Try loading weights only and rebuild model
    print("  Trying: Weights-only loading...")
    try:
        _cached_model = _load_model_weights_only()
        if _cached_model:
            print("✓ Model loaded with weights-only approach")
            return _cached_model
    except Exception as e:
        print(f"  ✗ Weights-only loading failed: {e}")
    
    print("✗ All loading strategies failed")
    return None


def _apply_compatibility_patches():
    """Apply monkey patches to handle Keras 3.x -> 2.x compatibility"""
    # Store original from_config methods
    original_input_layer_from_config = tf.keras.layers.InputLayer.from_config
    
    @classmethod
    def patched_input_layer_from_config(cls, config):
        # Handle batch_shape -> input_shape conversion
        if 'batch_shape' in config:
            batch_shape = config.pop('batch_shape')
            if batch_shape and len(batch_shape) > 1:
                config['input_shape'] = tuple(batch_shape[1:])
        
        # Handle DTypePolicy
        if 'dtype' in config and isinstance(config['dtype'], dict):
            dtype_obj = config['dtype']
            if dtype_obj.get('class_name') == 'DTypePolicy':
                config['dtype'] = dtype_obj.get('config', {}).get('name', 'float32')
        
        return original_input_layer_from_config.__func__(cls, config)
    
    # Apply patch
    tf.keras.layers.InputLayer.from_config = patched_input_layer_from_config
    
    # Patch all layers to handle dtype
    for layer_name in dir(tf.keras.layers):
        layer_cls = getattr(tf.keras.layers, layer_name)
        if isinstance(layer_cls, type) and issubclass(layer_cls, tf.keras.layers.Layer):
            try:
                original_from_config = layer_cls.from_config
                
                @classmethod
                def make_patched_from_config(orig_func):
                    def patched_from_config(cls, config):
                        # Handle DTypePolicy in dtype field
                        if 'dtype' in config and isinstance(config['dtype'], dict):
                            dtype_obj = config['dtype']
                            if dtype_obj.get('class_name') == 'DTypePolicy':
                                config['dtype'] = dtype_obj.get('config', {}).get('name', 'float32')
                        return orig_func.__func__(cls, config)
                    return patched_from_config
                
                # Only patch if not already patched
                if not hasattr(layer_cls.from_config, '_patched'):
                    layer_cls.from_config = make_patched_from_config(original_from_config)
                    layer_cls.from_config._patched = True
            except:
                pass


def _load_model_weights_only():
    """Load model by reconstructing architecture and loading weights"""
    
    # Try to extract and fix the model config, then rebuild
    with h5py.File(MODEL_PATH, 'r') as f:
        if 'model_config' not in f.attrs:
            return None
        
        model_config_str = f.attrs['model_config']
        if isinstance(model_config_str, bytes):
            model_config_str = model_config_str.decode('utf-8')
        
        model_config = json.loads(model_config_str)
        
        # Recursively fix all compatibility issues - remove dtype entirely
        def deep_fix(obj, parent_key=None):
            if isinstance(obj, dict):
                # Remove DTypePolicy entirely
                if obj.get('class_name') == 'DTypePolicy':
                    return None
                
                # Fix InputLayer batch_shape
                if obj.get('class_name') == 'InputLayer' and 'config' in obj:
                    if 'batch_shape' in obj['config']:
                        batch_shape = obj['config'].pop('batch_shape')
                        if batch_shape and len(batch_shape) > 1:
                            obj['config']['input_shape'] = list(batch_shape[1:])
                
                # Remove complex dtype from all layer configs
                if 'config' in obj and isinstance(obj['config'], dict):
                    if 'dtype' in obj['config'] and isinstance(obj['config']['dtype'], dict):
                        del obj['config']['dtype']
                
                result = {}
                for k, v in obj.items():
                    fixed_v = deep_fix(v, k)
                    # Skip dtype if it's None (was a DTypePolicy)
                    if k == 'dtype' and fixed_v is None:
                        continue
                    result[k] = fixed_v if fixed_v is not None else v
                return result
            elif isinstance(obj, list):
                return [deep_fix(item) for item in obj]
            return obj
        
        fixed_config = deep_fix(model_config)
    
    # Create a fixed model file
    temp_path = MODEL_PATH + '.fixed'
    shutil.copy2(MODEL_PATH, temp_path)
    
    with h5py.File(temp_path, 'r+') as f:
        f.attrs['model_config'] = json.dumps(fixed_config).encode('utf-8')
    
    try:
        model = tf.keras.models.load_model(temp_path, compile=False)
        os.remove(temp_path)
        return model
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def preprocess_image(image_path):
    """Preprocess image for prediction"""
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        return None
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"✗ Failed to read image: {image_path}")
        return None
    
    # Resize to model input size
    img = cv2.resize(img, IMAGE_SIZE)
    
    # Normalize pixel values to [0, 1]
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def predict(image_path):
    """
    Make prediction on an image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Prediction results with class, confidence, and all probabilities
    """
    print(f"\n{'='*50}")
    print(f"🔍 Predicting: {image_path}")
    print(f"{'='*50}")
    
    # Load model (cached after first load)
    model = load_model()
    if model is None:
        return None
    
    # Preprocess image
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None
    
    # Make prediction
    print("🧠 Running prediction...")
    predictions = model.predict(processed_image, verbose=0)
    
    # Extract results
    predicted_idx = np.argmax(predictions[0])
    predicted_class = CLASS_LABELS[predicted_idx]
    confidence = float(predictions[0][predicted_idx]) * 100
    
    # All class probabilities
    all_predictions = {
        label: round(float(predictions[0][i]) * 100, 2)
        for i, label in enumerate(CLASS_LABELS)
    }
    
    result = {
        'image_path': image_path,
        'predicted_class': predicted_class,
        'confidence': round(confidence, 2),
        'all_predictions': all_predictions
    }
    
    # Print results
    print(f"\n{'='*50}")
    print(f"📊 PREDICTION RESULTS")
    print(f"{'='*50}")
    print(f"  Image: {os.path.basename(image_path)}")
    print(f"  Predicted Class: {predicted_class}")
    print(f"  Confidence: {confidence:.2f}%")
    print(f"\n  All Probabilities:")
    for label, prob in all_predictions.items():
        bar = '█' * int(prob / 5) + '░' * (20 - int(prob / 5))
        print(f"    {label:10s}: {bar} {prob:.2f}%")
    print(f"{'='*50}\n")
    
    return result


def predict_multiple(image_paths):
    """Predict on multiple images"""
    results = []
    for path in image_paths:
        result = predict(path)
        if result:
            results.append(result)
    return results


# ============================================
# Main - Example Usage
# ============================================
if __name__ == '__main__':
    import sys
    
    print("\n" + "="*60)
    print("🔬 BREAST CANCER DETECTION - LOCAL PREDICTION")
    print("="*60)
    
    if len(sys.argv) > 1:
        # Use command line arguments
        image_paths = sys.argv[1:]
    else:
        # Default: Ask for image path
        print("\nUsage:")
        print("  python predict_local.py <image_path>")
        print("  python predict_local.py image1.jpg image2.png ...")
        print("\nExample:")
        print("  python predict_local.py ~/Desktop/ultrasound.jpg")
        print("\n" + "-"*60)
        
        # Interactive mode
        image_path = input("\nEnter image path (or 'q' to quit): ").strip()
        
        if image_path.lower() == 'q':
            print("Goodbye!")
            sys.exit(0)
        
        # Handle drag-and-drop paths (may have quotes)
        image_path = image_path.strip('"').strip("'")
        
        if not image_path:
            print("No image path provided. Exiting.")
            sys.exit(1)
        
        image_paths = [image_path]
    
    # Run predictions
    for path in image_paths:
        # Handle paths with spaces or special characters
        path = path.strip('"').strip("'")
        result = predict(path)
        
        if result:
            print(f"✓ Prediction complete for: {os.path.basename(path)}")
        else:
            print(f"✗ Prediction failed for: {path}")
