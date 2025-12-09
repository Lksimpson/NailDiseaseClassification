# Nail Disease Classification

A comprehensive deep learning system for detecting and classifying nail diseases using multiple state-of-the-art computer vision models with an interactive web interface.

## Overview

This project implements a multi-model approach to nail disease classification, combining object detection and classification techniques to provide accurate diagnosis of various nail conditions. The system includes five different classification models, YOLO-based nail detection, GradCAM visualization for model interpretability, and a user-friendly web interface.

## Features

- **Multiple Classification Models**: Five different CNN architectures for robust disease classification
  - AlexNet
  - VGG16
  - ResNet50
  - DenseNet
  - GoogleNet
- **Object Detection**: YOLO-based nail region detection
- **Model Interpretability**: GradCAM visualization to understand model predictions
- **Web Interface**: Interactive Flask-based UI for easy image upload and analysis
- **Real-time Predictions**: Quick disease classification with confidence scores

## Project Structure

```
NailDiseaseClassification/
├── models/                          # Trained model weights
├── static/                          # Static assets for web interface
├── templates/                       # HTML templates for Flask app
├── runs/nail-detector2/             # YOLO training outputs
├── alexnet-nailclassification.ipynb
├── vgg16-nailclassification.ipynb
├── resnet50-nail-classification.ipynb
├── densenet-10-epoch-and-32-batch.ipynb
├── GoogleNet.ipynb
├── yolo_nail_object_detection.ipynb
├── app.py                           # Flask web application
├── best.pt                          # Best YOLO model weights
├── requirements.txt                 # Python dependencies
├── MODEL_SETUP.md                   # Model setup instructions
└── README_APP.md                    # Application documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Lksimpson/NailDiseaseClassification.git
cd NailDiseaseClassification
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

Each model has its own Jupyter notebook for training:

- **AlexNet**: `alexnet-nailclassification.ipynb`
- **VGG16**: `vgg16-nailclassification.ipynb`
- **ResNet50**: `resnet50-nail-classification.ipynb`
- **DenseNet**: `densenet-10-epoch-and-32-batch.ipynb`
- **GoogleNet**: `GoogleNet.ipynb`
- **YOLO Detection**: `yolo_nail_object_detection.ipynb`

Open any notebook in Jupyter and follow the training steps to train models on your dataset.

### Running the Web Application

Start the Flask web interface:

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Using the Application

1. Navigate to the web interface
2. Upload an image of a nail
3. The system will:
   - Detect the nail region using YOLO
   - Classify the nail disease using multiple models
   - Display predictions with confidence scores
   - Show GradCAM visualizations to explain the predictions

## Models

### Classification Models

All five classification models are trained on the same nail disease dataset:

- **AlexNet**
- **VGG16**
- **ResNet50**
- **DenseNet**
- **GoogleNet**

### Object Detection

- **YOLOv8**: Real-time nail detection to localize nail regions before classification

### Model Interpretability

- **GradCAM**: Generates heatmaps showing which regions of the image the model focuses on for its predictions

## Dataset

The models are trained on a nail disease dataset containing images of various nail conditions. 

## Model Performance

Each model notebook contains:
- Training and validation accuracy curves
- Confusion matrices
- Classification reports
- Sample predictions

Refer to individual notebooks for detailed performance metrics.

## API Endpoints

The Flask application provides the following endpoints:

- `GET /`: Main page with upload interface
- `POST /predict`: Upload image and get predictions
- `GET /visualize`: View GradCAM visualizations

See `README_APP.md` for detailed API documentation.
