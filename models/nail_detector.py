"""
YOLO-based nail detection module
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os


class NailDetector:
    """YOLO-based nail detector"""
    
    def __init__(self, model_path=None, conf_threshold=0.25):
        """
        Initialize the nail detector
        
        Args:
            model_path: Path to YOLO model weights. If None, uses default YOLOv8n
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # Use provided model or default YOLOv8n
        if model_path is None or not os.path.exists(model_path):
            # Use pretrained YOLOv8n (general object detection)
            # In production, you'd use a custom-trained nail detection model
            self.model = YOLO('yolov8n.pt')
        else:
            self.model = YOLO(model_path)
    
    def detect_nails(self, image):
        """
        Detect nails in an image
        
        Args:
            image: Input image (BGR format, numpy array)
            
        Returns:
            dict with keys:
                - 'nails': List of detected nails, each with 'crop', 'bbox', 'confidence'
                - 'visualization': Image with bounding boxes drawn
        """
        # Run YOLO inference
        results = self.model(image, conf=self.conf_threshold)[0]
        boxes = results.boxes
        
        if len(boxes) == 0:
            return None
        
        # Extract all detections
        nails = []
        h, w = image.shape[:2]
        vis_image = image.copy()
        
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf.cpu().numpy())
            
            # Clamp to image bounds
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop nail region
            nail_crop = image[y1:y2, x1:x2].copy()
            
            # Store detection info
            nails.append({
                'crop': nail_crop,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence
            })
            
            # Draw bounding box on visualization
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Nail {confidence:.2f}"
            cv2.putText(
                vis_image,
                label,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        
        return {
            'nails': nails,
            'visualization': vis_image
        }

