# Nail Disease Screening Application

A full end-to-end machine learning web application for nail disease screening (not a medical diagnosis tool).

## ⚠️ Medical Disclaimer

**This application is for informational purposes only and is not a substitute for professional medical diagnosis, treatment, or advice.** Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.

## Features

- 🖼️ **Image Upload**: Upload hand images via web interface
- 🔍 **Nail Detection**: YOLO-based object detection to identify and crop nail regions
- 🧠 **Disease Classification**: Deep learning model to predict potential nail conditions
- 📊 **Results Display**: Probability scores and localization overlays
- ⚠️ **Medical Disclaimer**: Clear disclaimer displayed on the interface

## Architecture

### Components

1. **Backend (Flask)**: `app.py` - Main web server and API endpoints
2. **Nail Detector**: `models/nail_detector.py` - YOLO-based nail detection
3. **Disease Classifier**: `models/disease_classifier.py` - DenseNet201-based classification
4. **Frontend**: HTML/CSS/JavaScript for user interface

### Disease Classes

The model can classify the following conditions:
- Acral Lentiginous Melanoma
- Healthy Nail
- Onychogryphosis
- Blue Finger
- Clubbing
- Pitting

## Setup Instructions

### 1. Activate Virtual Environment

```bash
cd Nail-Disease
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Model Weights

**Important**: The application requires trained DenseNet model weights for accurate predictions.

**Option A: Use Trained Model**
- Train your model using `densenet_test_disease_classification.ipynb`
- Save the model weights (see `MODEL_SETUP.md` for details)
- Place the `.pth` file in the `models/` directory with one of these names:
  - `densenet_nail_disease_best.pth` (recommended)
  - `densenet_nail_disease_weights.pth`
  - `densenet_model.pth`
  - `nail_disease_classifier.pth`

**Option B: Use Environment Variable**
```bash
export DISEASE_MODEL_PATH=/path/to/your/model.pth
```

**Note**: If no trained weights are found, the app will use randomly initialized weights (not suitable for predictions). The app will automatically search for model files and display a message indicating whether weights were loaded.

For YOLO: The application will automatically download YOLOv8n for nail detection.

### 4. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. Open the web interface in your browser
2. Upload a hand image (PNG, JPG, JPEG, GIF, WEBP, max 16MB)
3. Click "Analyze Image"
4. View the results:
   - Detection visualization showing detected nail regions
   - Classification results with probability scores
   - All condition probabilities for each detected nail

## Project Structure

```
Nail-Disease/
├── app.py                      # Main Flask application
├── models/
│   ├── __init__.py
│   ├── nail_detector.py        # YOLO nail detection
│   └── disease_classifier.py   # Disease classification
├── templates/
│   └── index.html              # Main web page
├── static/
│   ├── css/
│   │   └── style.css           # Styling
│   └── js/
│       └── app.js              # Frontend JavaScript
├── uploads/                     # Uploaded images (created automatically)
├── results/                     # Results storage (created automatically)
├── requirements.txt             # Python dependencies
└── README_APP.md               # This file
```

## API Endpoints

### `GET /`
Main page with upload interface

### `GET /health`
Health check endpoint

### `POST /predict`
Main prediction endpoint
- **Input**: Multipart form data with `image` field
- **Output**: JSON with detection and classification results

## Notes

- **Model Weights Required**: The DenseNet classifier requires trained model weights to make accurate predictions. See `MODEL_SETUP.md` for instructions on how to save and load trained models.
- The YOLO model uses a general-purpose pretrained model. For better nail detection, train a custom YOLO model on nail-specific data.
- The disease classifier uses DenseNet201 architecture. You must train the model on your dataset and save the weights for production use.
- The application is designed for screening purposes only, not medical diagnosis.

## Development

To improve the application:

1. Train a custom YOLO model for better nail detection
2. Train the DenseNet201 classifier on your dataset
3. Add model persistence and caching
4. Add user authentication and history
5. Improve error handling and validation

