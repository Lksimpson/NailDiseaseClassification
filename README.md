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

## Dataset
The models are trained on nail disease datasets containing images of various nail conditions. The datasets can be accessed from multiple sources:
Classification Dataset

Kaggle Nail Disease Dataset: https://www.kaggle.com/datasets/nikhilgurav21/nail-disease-detection-dataset/data 

Contains labeled images of different nail diseases
Used for training AlexNet, VGG16, ResNet50, DenseNet, and GoogleNet models



## Object Detection Dataset

Roboflow Fingernails Dataset: https://universe.roboflow.com/fingernail-ztwys/fingernails-xb812

Contains annotated nail images for object detection
Used for training the YOLO nail detection model



Setup Instructions
For Classification Models (Kaggle):

Download the dataset from Kaggle
Extract the files to a data/ directory in the project root
Ensure the dataset is organized with separate folders for each disease class
Update the dataset paths in the respective notebook files in the `notebooks/` directory before training
Note: You may need to create a Kaggle account and accept the dataset's terms of use

For YOLO Detection (Roboflow):

Access the Roboflow dataset at the link above
Generate and download the dataset in YOLO format
Place the dataset in the appropriate directory structure
Update the data configuration in `notebooks/yolo_nail_object_detection.ipynb`
The dataset includes train, validation, and test splits

## Project Structure

```
NailDiseaseClassification/
├── notebooks/                       # Jupyter notebooks for model training
│   ├── alexnet-nailclassification.ipynb
│   ├── vgg16-nailclassification.ipynb
│   ├── resnet50-nail-classification (1).ipynb
│   ├── densenet-10-epoch-and-32-batch.ipynb
│   ├── GoogleNet.ipynb
│   ├── GoogleNet.html              # Exported HTML version
│   └── yolo_nail_object_detection.ipynb
├── models/                          # Model code and trained weights
│   ├── __init__.py
│   ├── disease_classifier.py       # Disease classification model
│   └── nail_detector.py            # YOLO nail detection model
├── data/                            # Dataset directory
│   ├── train/                      # Training images
│   ├── valid/                      # Validation images
│   └── test/                       # Test images
├── static/                          # Static assets for web interface
│   ├── css/
│   └── js/
├── templates/                       # HTML templates for Flask app
├── runs/                            # Training outputs
│   └── nail-detector2/             # YOLO training outputs
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

Each model has its own Jupyter notebook for training, located in the `notebooks/` directory:

- **AlexNet**: `notebooks/alexnet-nailclassification.ipynb`
- **VGG16**: `notebooks/vgg16-nailclassification.ipynb`
- **ResNet50**: `notebooks/resnet50-nail-classification (1).ipynb`
- **DenseNet**: `notebooks/densenet-10-epoch-and-32-batch.ipynb`
- **GoogleNet**: `notebooks/GoogleNet.ipynb`
- **YOLO Detection**: `notebooks/yolo_nail_object_detection.ipynb`

Open any notebook in Jupyter and follow the training steps to train models on your dataset. All notebooks are organized in the `notebooks/` folder for better project structure.

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
