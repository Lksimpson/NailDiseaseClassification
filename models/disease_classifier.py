"""
Disease classification module for nail conditions
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from collections import OrderedDict


class DiseaseClassifier:
    """Disease classifier for nail conditions"""
    
    # Disease classes (matching the training data)
    CLASSES = [
        'Acral_Lentiginous_Melanoma',
        'Healthy_Nail',
        'Onychogryphosis',
        'blue_finger',
        'clubbing',
        'pitting'
    ]
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the disease classifier
        
        Args:
            model_path: Path to trained model weights. If None, uses pretrained DenseNet201
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = len(self.CLASSES)
        
        # Initialize model architecture
        self.model = self._create_model()
        
        # Load weights if provided
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                # Handle both checkpoint format and state_dict format
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model checkpoint from {model_path}")
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                    print(f"Loaded model checkpoint from {model_path}")
                else:
                    # Assume it's a state_dict directly
                    self.model.load_state_dict(checkpoint)
                    print(f"Loaded model weights from {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load model weights from {model_path}: {e}")
                print("Using randomly initialized model.")
        else:
            if model_path:
                print(f"Warning: Model path specified but file not found: {model_path}")
            print("Warning: No model weights provided. Using randomly initialized model.")
            print("For production use, please train and save a model first.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _create_model(self):
        """Create DenseNet201 model with custom classifier"""
        # Load pretrained DenseNet201
        model = models.densenet201(weights='DEFAULT')
        
        # Freeze feature extractor
        for param in model.parameters():
            param.requires_grad = False
        
        # Custom classifier (matching training architecture)
        hidden_units = 1024
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1920, hidden_units)),  # DenseNet201 features output 1920
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(hidden_units, 512)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.3)),
            ('fc3', nn.Linear(512, self.num_classes)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        
        model.classifier = classifier
        return model
    
    def classify(self, nail_image):
        """
        Classify a nail image
        
        Args:
            nail_image: Nail image (BGR numpy array from OpenCV)
            
        Returns:
            dict with keys:
                - 'predicted_class': Name of predicted class
                - 'probability': Confidence score
                - 'all_probabilities': Dict of all class probabilities
        """
        # Convert BGR to RGB
        if len(nail_image.shape) == 3 and nail_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(nail_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = nail_image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            # Convert LogSoftmax to probabilities
            probabilities = torch.exp(output).cpu().numpy()[0]
        
        # Get predicted class
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.CLASSES[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Create probability dictionary
        all_probs = {
            self.CLASSES[i]: float(probabilities[i])
            for i in range(len(self.CLASSES))
        }
        
        return {
            'predicted_class': predicted_class,
            'probability': confidence,
            'all_probabilities': all_probs
        }

