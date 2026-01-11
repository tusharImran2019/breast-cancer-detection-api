from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

try:
    model = tf.keras.models.load_model('breast_cancer_3modality_fusion.h5')
    print("Model loaded successfully!")
except:
    print("Model file not found")
    model = None

CLASS_LABELS = ['Benign', 'Malignant', 'Normal']

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'message': 'Breast Cancer Detection API v1.0',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        processed_image = preprocess_image(image_bytes)
        
        predictions = model.predict(processed_image, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASS_LABELS[predicted_idx]
        confidence = float(predictions[0][predicted_idx]) * 100
        
        all_predictions = {
            'Benign': float(predictions[0][0]) * 100,
            'Malignant': float(predictions[0][1]) * 100,
            'Normal': float(predictions[0][2]) * 100
        }
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'all_predictions': all_predictions,
            'message': f'Detection: {predicted_class} ({confidence:.2f}% confidence)'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
