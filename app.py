"""
Nail Disease Screening Application
A web application for nail disease classification (not a medical diagnosis tool)
"""

import os
import io
import base64
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image

from models.nail_detector import NailDetector
from models.disease_classifier import DiseaseClassifier

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Model configuration
MODELS_DIR = 'models'
DISEASE_MODEL_PATH = os.getenv('DISEASE_MODEL_PATH', None)

# Try to find trained model weights if not specified
if DISEASE_MODEL_PATH is None:
    # Check common locations for trained models
    possible_paths = [
        os.path.join(MODELS_DIR, 'densenet_nail_disease_best.pth'),
        os.path.join(MODELS_DIR, 'densenet_nail_disease_weights.pth'),
        os.path.join(MODELS_DIR, 'densenet_model.pth'),
        os.path.join(MODELS_DIR, 'nail_disease_classifier.pth'),
        'densenet_nail_disease_best.pth',
        'densenet_nail_disease_weights.pth',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            DISEASE_MODEL_PATH = path
            print(f"Found model weights at: {path}")
            break

# Initialize models
print("Loading models...")
nail_detector = NailDetector()
disease_classifier = DiseaseClassifier(model_path=DISEASE_MODEL_PATH)
print("Models loaded successfully!")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image_to_base64(image_path):
    """Encode image to base64 string for frontend display"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'models_loaded': True})


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image (png, jpg, jpeg, gif, webp)'}), 400
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Step 1: Detect nails using YOLO
        print("Detecting nails...")
        detection_results = nail_detector.detect_nails(image)
        
        if not detection_results or len(detection_results['nails']) == 0:
            return jsonify({
                'error': 'No nails detected in the image. Please upload an image with visible nails.',
                'detection_visualization': None
            }), 400
        
        # Step 2: Classify each detected nail
        print("Classifying nail diseases...")
        all_results = []
        
        for idx, nail_data in enumerate(detection_results['nails']):
            nail_crop = nail_data['crop']
            bbox = nail_data['bbox']
            confidence = nail_data['confidence']
            
            # Classify the nail crop
            classification_result = disease_classifier.classify(nail_crop)
            
            all_results.append({
                'nail_index': idx + 1,
                'bbox': bbox,
                'detection_confidence': float(confidence),
                'disease': classification_result['predicted_class'],
                'probability': float(classification_result['probability']),
                'all_probabilities': classification_result['all_probabilities']
            })
        
        # Encode visualization image
        vis_image_base64 = None
        if detection_results.get('visualization') is not None:
            # Save visualization temporarily
            vis_path = os.path.join(app.config['RESULTS_FOLDER'], 'temp_vis.jpg')
            cv2.imwrite(vis_path, detection_results['visualization'])
            vis_image_base64 = encode_image_to_base64(vis_path)
            os.remove(vis_path)  # Clean up
        
        return jsonify({
            'success': True,
            'num_nails_detected': len(all_results),
            'results': all_results,
            'detection_visualization': vis_image_base64
        })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5001))
    print(f"Starting Flask server on port {port}...")
    app.run(debug=True, host='0.0.0.0', port=port)

