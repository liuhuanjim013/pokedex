#!/usr/bin/env python3
"""
Model Performance Comparison Script
Compares performance between the best.pt model and the ONNX model to ensure conversion quality.
"""

import os
import sys
import argparse
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import onnxruntime as ort
from ultralytics import YOLO

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelPerformanceComparator:
    """Compares performance between PyTorch and ONNX models."""
    
    def __init__(self, pt_model_path: str, onnx_model_path: str, classes_path: str):
        """
        Initialize comparator.
        
        Args:
            pt_model_path: Path to best.pt model
            onnx_model_path: Path to ONNX model
            classes_path: Path to classes.txt file
        """
        self.pt_model_path = pt_model_path
        self.onnx_model_path = onnx_model_path
        self.classes_path = classes_path
        self.class_names = self._load_class_names()
        self.num_classes = len(self.class_names)
        
        # Load models
        self.pt_model = self._load_pt_model()
        self.onnx_session = self._load_onnx_model()
        
        # Performance tracking
        self.pt_results = []
        self.onnx_results = []
        
        logger.info(f"Initialized comparator with {self.num_classes} classes")
    
    def _load_class_names(self) -> List[str]:
        """Load Pokemon class names."""
        try:
            with open(self.classes_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(class_names)} class names")
            return class_names
        except Exception as e:
            logger.error(f"Failed to load class names: {e}")
            return [f"pokemon_{i}" for i in range(1025)]
    
    def _load_pt_model(self):
        """Load PyTorch model."""
        try:
            logger.info(f"Loading PyTorch model: {self.pt_model_path}")
            model = YOLO(self.pt_model_path)
            
            # Debug: Check model type and configuration
            logger.info(f"Model type: {type(model)}")
            logger.info(f"Model task: {getattr(model, 'task', 'unknown')}")
            
            # Check if it's a classification model
            if hasattr(model, 'model') and hasattr(model.model, 'names'):
                logger.info(f"Model has {len(model.model.names)} classes")
                logger.info(f"First few class names: {list(model.model.names.values())[:5]}")
            
            logger.info("PyTorch model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            return None
    
    def _load_onnx_model(self):
        """Load ONNX model."""
        try:
            logger.info(f"Loading ONNX model: {self.onnx_model_path}")
            session = ort.InferenceSession(self.onnx_model_path)
            
            # Debug: Check ONNX model inputs and outputs
            input_details = session.get_inputs()
            output_details = session.get_outputs()
            
            logger.info(f"ONNX inputs: {[inp.name for inp in input_details]}")
            logger.info(f"ONNX outputs: {[out.name for out in output_details]}")
            
            for inp in input_details:
                logger.info(f"Input '{inp.name}' shape: {inp.shape}")
            
            for out in output_details:
                logger.info(f"Output '{out.name}' shape: {out.shape}")
            
            logger.info("ONNX model loaded successfully")
            return session
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return None
    
    def preprocess_image(self, image) -> np.ndarray:
        """
        Preprocess image for ONNX model input.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Convert PIL to numpy if needed
            if hasattr(image, 'convert'):
                image = np.array(image.convert('RGB'))
            
            # Resize to 256x256
            if image.shape[:2] != (256, 256):
                import cv2
                image = cv2.resize(image, (256, 256))
            
            # Normalize to [0, 1] and convert to float32
            image = image.astype(np.float32) / 255.0
            
            # Convert to NCHW format
            image = np.transpose(image, (2, 0, 1))
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            return None
    
    def predict_pt_model(self, image) -> Tuple[int, float, float]:
        """
        Run inference with PyTorch model.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (predicted_class, confidence, inference_time)
        """
        try:
            if self.pt_model is None:
                return 0, 0.0, 0.0
            
            # Run inference
            start_time = time.time()
            results = self.pt_model(image, verbose=False)
            inference_time = time.time() - start_time
            
            # Debug logging for first few predictions
            if len(self.pt_results) < 3:  # Only log first 3 predictions
                logger.info(f"PyTorch Debug - Sample {len(self.pt_results)}:")
                logger.info(f"  Results type: {type(results)}")
                logger.info(f"  Results length: {len(results) if isinstance(results, list) else 'N/A'}")
                
                if isinstance(results, list) and len(results) > 0:
                    result = results[0]  # Get the first result
                    logger.info(f"  First result type: {type(result)}")
                    logger.info(f"  First result attributes: {dir(result)}")
                    
                    # Check if result has boxes
                    if hasattr(result, 'boxes'):
                        logger.info(f"  Has boxes: {result.boxes is not None}")
                        if result.boxes is not None:
                            logger.info(f"  Boxes length: {len(result.boxes)}")
                            if len(result.boxes) > 0:
                                logger.info(f"  Boxes shape: {result.boxes.xyxy.shape if hasattr(result.boxes, 'xyxy') else 'N/A'}")
                                logger.info(f"  Boxes classes: {result.boxes.cls}")
                                logger.info(f"  Boxes confidences: {result.boxes.conf}")
                            else:
                                logger.info(f"  No detections found")
                    else:
                        logger.info(f"  No boxes attribute")
                    
                    # Check if result has probs
                    if hasattr(result, 'probs'):
                        logger.info(f"  Has probs: {result.probs is not None}")
                        if result.probs is not None:
                            logger.info(f"  Probs shape: {result.probs.shape if hasattr(result.probs, 'shape') else 'N/A'}")
                            logger.info(f"  Probs values: {result.probs}")
                    else:
                        logger.info(f"  No probs attribute")
                else:
                    logger.info(f"  No results found")
            
            # Handle list of results
            if isinstance(results, list) and len(results) > 0:
                result = results[0]  # Get the first result
                
                # Extract the class and confidence from the top detection
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    # Get the first (best) detection
                    boxes = result.boxes
                    if len(boxes) > 0:
                        # Get class and confidence from the first detection
                        predicted_class = int(boxes.cls[0].item())
                        confidence = float(boxes.conf[0].item())
                        
                        if len(self.pt_results) < 3:  # Only log first 3 predictions
                            logger.info(f"  Predicted class: {predicted_class}")
                            logger.info(f"  Confidence: {confidence:.4f}")
                            logger.info(f"  Number of detections: {len(boxes)}")
                        
                        return predicted_class, confidence, inference_time
                
                # Fallback: try to get classification result
                if hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs
                    if hasattr(probs, 'shape') and len(probs.shape) > 0:
                        predicted_class = int(probs.argmax().item())
                        confidence = float(probs.max().item())
                        
                        if len(self.pt_results) < 3:  # Only log first 3 predictions
                            logger.info(f"  Using probs - Predicted class: {predicted_class}")
                            logger.info(f"  Using probs - Confidence: {confidence:.4f}")
                        
                        return predicted_class, confidence, inference_time
            
            # If we get here, no valid prediction was found
            if len(self.pt_results) < 3:  # Only log first 3 predictions
                logger.warning(f"  No valid prediction found - returning default values")
            
            return 0, 0.0, inference_time
            
        except Exception as e:
            logger.error(f"Failed to predict with PyTorch model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0, 0.0, 0.0
    
    def predict_onnx_model(self, image) -> Tuple[int, float, float]:
        """
        Run inference with ONNX model.
        Correctly interpret detection output for classification.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (predicted_class, confidence, inference_time)
        """
        try:
            if self.onnx_session is None:
                return 0, 0.0, 0.0
            
            # Preprocess image
            input_data = self.preprocess_image(image)
            if input_data is None:
                return 0, 0.0, 0.0
            
            # Run inference
            start_time = time.time()
            outputs = self.onnx_session.run(['output0'], {'images': input_data})
            inference_time = time.time() - start_time
            
            # Process output - ONNX output shape: [1, 1029, 1344]
            # 1029 = 1025 classes + 4 bounding box coordinates (x, y, w, h)
            # 1344 = number of detection boxes
            detection_output = outputs[0][0]  # Shape: [1029, 1344]
            
            # Try transposed interpretation - maybe the output is [num_boxes, classes+bbox] instead of [classes+bbox, num_boxes]
            # The pattern suggests indices 0-3 have large values, which might be bbox coords
            detection_output_transposed = detection_output.T  # Shape: [1344, 1029]
            
            # Extract class logits from the last 1025 values of each detection box (after bbox coords)
            class_logits = detection_output_transposed[:, 4:1029]  # Shape: [1344, 1025]
            
            # Find the detection box with the highest confidence for any class
            max_logits = np.max(class_logits, axis=1)  # Shape: [1344]
            best_box_idx = np.argmax(max_logits)
            
            # Get the class logits for the best detection box
            best_box_logits = class_logits[best_box_idx, :]  # Shape: [1025]
            
            # Apply softmax to get class probabilities
            class_probs = self._softmax(best_box_logits)
            
            # Get the predicted class and confidence
            predicted_class = np.argmax(class_probs)
            confidence = class_probs[predicted_class]
            
            # Debug logging for first few predictions
            if len(self.onnx_results) < 3:  # Only log first 3 predictions
                logger.info(f"ONNX Debug - Sample {len(self.onnx_results)}:")
                logger.info(f"  Detection output shape: {detection_output.shape}")
                logger.info(f"  Class logits shape: {class_logits.shape}")
                logger.info(f"  Best box index: {best_box_idx}")
                logger.info(f"  Max logit across all boxes: {np.max(max_logits):.4f}")
                logger.info(f"  Predicted class: {predicted_class}")
                logger.info(f"  Confidence: {confidence:.4f}")
                logger.info(f"  Top 5 class probabilities: {np.argsort(class_probs)[-5:][::-1]}")
                logger.info(f"  Top 5 confidence values: {np.sort(class_probs)[-5:][::-1]}")
                
                # Additional debug: Check raw output values
                logger.info(f"  Raw detection output range: [{np.min(detection_output):.4f}, {np.max(detection_output):.4f}]")
                logger.info(f"  Raw class logits range: [{np.min(class_logits):.4f}, {np.max(class_logits):.4f}]")
                logger.info(f"  Raw class logits mean: {np.mean(class_logits):.4f}")
                logger.info(f"  Raw class logits std: {np.std(class_logits):.4f}")
                
                # Check if all values are the same (indicates a problem)
                if np.allclose(class_logits, class_logits[0, 0]):
                    logger.warning(f"  WARNING: All class logits are identical! This indicates a problem with the ONNX model.")
                
                # Show detailed analysis of all 1029 values for the best box
                logger.info(f"  Detailed analysis of best box (index {best_box_idx}):")
                logger.info(f"    All 1029 values: {detection_output[:, best_box_idx]}")
                
                # Show first 20 and last 20 values
                first_20 = detection_output[:20, best_box_idx]
                last_20 = detection_output[-20:, best_box_idx]
                logger.info(f"    First 20 values: {first_20}")
                logger.info(f"    Last 20 values: {last_20}")
                
                # Show non-zero values
                non_zero_mask = detection_output[:, best_box_idx] != 0
                non_zero_indices = np.where(non_zero_mask)[0]
                non_zero_values = detection_output[non_zero_indices, best_box_idx]
                logger.info(f"    Non-zero values count: {len(non_zero_indices)}")
                logger.info(f"    Non-zero indices: {non_zero_indices}")
                logger.info(f"    Non-zero values: {non_zero_values}")
                
                # Show pattern analysis
                logger.info(f"    Value distribution:")
                logger.info(f"      - Zeros: {np.sum(detection_output[:, best_box_idx] == 0)}")
                logger.info(f"      - Positive: {np.sum(detection_output[:, best_box_idx] > 0)}")
                logger.info(f"      - Negative: {np.sum(detection_output[:, best_box_idx] < 0)}")
                logger.info(f"      - Unique values: {len(np.unique(detection_output[:, best_box_idx]))}")
                
                # Show if there's a pattern in the first 1025 vs last 4 values
                first_1025 = detection_output[:1025, best_box_idx]
                last_4 = detection_output[1025:, best_box_idx]
                logger.info(f"    First 1025 values (classes): min={np.min(first_1025):.4f}, max={np.max(first_1025):.4f}, mean={np.mean(first_1025):.4f}")
                logger.info(f"    Last 4 values (bbox): {last_4}")
                
                # Show if the pattern is consistent across different boxes
                if len(self.onnx_results) == 1:  # Only for first sample to avoid too much output
                    logger.info(f"  Pattern analysis across different boxes:")
                    for box_idx in [0, 100, 500, 1000, 1333]:  # Sample different boxes
                        box_values = detection_output[:, box_idx]
                        max_val_idx = np.argmax(box_values)
                        max_val = box_values[max_val_idx]
                        logger.info(f"    Box {box_idx}: max value {max_val:.4f} at index {max_val_idx}")
            
            # Ensure predicted class is within bounds
            if predicted_class >= self.num_classes:
                predicted_class = self.num_classes - 1
            
            return int(predicted_class), float(confidence), inference_time
            
        except Exception as e:
            logger.error(f"Failed to predict with ONNX model: {e}")
            return 0, 0.0, 0.0
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def load_validation_data(self) -> List[Tuple[Any, int]]:
        """
        Load validation data using YOLO framework's built-in data loading.
        This matches the approach used in the training script.
        
        Returns:
            List of (image, class_id) tuples
        """
        try:
            from PIL import Image
            import tempfile
            import yaml
            
            # Create a temporary data.yaml file for YOLO framework
            data_yaml_content = f"""# Temporary data configuration for validation
path: {Path(__file__).resolve().parents[3] / "data" / "processed" / "yolo_dataset"}
train: train/images
val: validation/images
test: test/images
nc: 1025
names:
"""
            
            # Add all 1025 Pokemon names (0-1024)
            for i in range(1025):
                data_yaml_content += f"  {i}: pokemon_{i}\n"
            
            # Create temporary data.yaml file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(data_yaml_content)
                temp_data_yaml = f.name
            
            try:
                # Use YOLO framework's built-in data loading
                from ultralytics.data import YOLODataset
                
                # Load validation dataset using YOLO framework
                dataset = YOLODataset(
                    img_path=str(Path(__file__).resolve().parents[3] / "data" / "processed" / "yolo_dataset" / "validation" / "images"),
                    data=temp_data_yaml,
                    task='detect'
                )
                
                validation_data = []
                logger.info(f"Loading {len(dataset)} validation samples using YOLO framework")
                
                for i in range(len(dataset)):
                    try:
                        # Get image and label from dataset
                        sample = dataset[i]
                        image = sample['img']  # This is already a PIL Image
                        
                        # Extract class ID from label
                        if 'labels' in sample and len(sample['labels']) > 0:
                            # Get the first label's class ID
                            class_id = int(sample['labels'][0][0])  # First label, first element is class ID
                            
                            # Ensure class_id is within bounds (0-1024 for 1025 classes)
                            if class_id > 1024:
                                class_id = 1024
                            elif class_id < 0:
                                class_id = 0
                            
                            validation_data.append((image, class_id))
                        else:
                            # If no labels, skip this image
                            continue
                            
                    except Exception as e:
                        logger.warning(f"Failed to load sample {i}: {e}")
                        continue
                
                logger.info(f"Successfully loaded {len(validation_data)} validation samples using YOLO framework")
                return validation_data
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_data_yaml)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Failed to load validation data using YOLO framework: {e}")
            logger.info("Falling back to manual filename parsing...")
            
            # Fallback to manual filename parsing (original method)
            return self._load_validation_data_manual()
    
    def _load_validation_data_manual(self) -> List[Tuple[Any, int]]:
        """
        Fallback method: Load validation data by manually parsing filenames.
        Maps Pokemon IDs to class IDs according to architecture:
        - Pokemon ID 0001 → class_id 0
        - Pokemon ID 1025 → class_id 1024
        """
        try:
            from PIL import Image
            import re
            
            # Load local validation dataset
            validation_dir = Path(__file__).resolve().parents[3] / "data" / "processed" / "yolo_dataset" / "validation"
            images_dir = validation_dir / "images"
            
            if not images_dir.exists():
                logger.error(f"Validation images directory not found: {images_dir}")
                return []
            
            validation_data = []
            image_files = list(images_dir.glob("*.jpg"))
            
            logger.info(f"Found {len(image_files)} validation images")
            
            for img_file in tqdm(image_files, desc="Loading validation data"):
                try:
                    # Load image
                    image = Image.open(img_file).convert('RGB')
                    
                    # Extract Pokemon ID from filename (format: pokemon_id_image.jpg)
                    filename = img_file.stem
                    
                    # Try different patterns for Pokemon ID extraction
                    pokemon_id = None
                    
                    # Pattern 1: 4-digit prefix (e.g., 0001_001.jpg)
                    match = re.match(r'^(\d{4})_\d+', filename)
                    if match:
                        pokemon_id = int(match.group(1))
                    else:
                        # Pattern 2: Just 4-digit number (e.g., 0001.jpg)
                        match = re.match(r'^(\d{4})', filename)
                        if match:
                            pokemon_id = int(match.group(1))
                        else:
                            # Pattern 3: Any number at start (e.g., 1_001.jpg)
                            match = re.match(r'^(\d+)', filename)
                            if match:
                                pokemon_id = int(match.group(1))
                    
                    if pokemon_id is None:
                        logger.warning(f"Could not extract Pokemon ID from filename: {filename}")
                        continue
                    
                    # Map Pokemon ID to class ID according to architecture:
                    # Pokemon ID 0001 → class_id 0
                    # Pokemon ID 1025 → class_id 1024
                    class_id = pokemon_id - 1
                    
                    # Ensure class_id is within bounds (0-1024 for 1025 classes)
                    if class_id > 1024:
                        logger.warning(f"Pokemon ID {pokemon_id} (class_id {class_id}) exceeds model range (0-1024)")
                        class_id = 1024
                    elif class_id < 0:
                        logger.warning(f"Pokemon ID {pokemon_id} (class_id {class_id}) is negative")
                        class_id = 0
                    
                    validation_data.append((image, class_id))
                    
                except Exception as e:
                    logger.warning(f"Failed to load {img_file}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {len(validation_data)} validation samples")
            return validation_data
            
        except Exception as e:
            logger.error(f"Failed to load validation data: {e}")
            return []
    
    def evaluate_model(self, validation_data: List[Tuple], max_samples: int = None) -> Dict[str, Any]:
        """
        Evaluate both models on validation set.
        
        Args:
            validation_data: List of (image, class_id) tuples
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Starting model comparison with {len(validation_data)} samples")
        
        if max_samples:
            validation_data = validation_data[:max_samples]
            logger.info(f"Limited to {max_samples} samples for evaluation")
        
        # Initialize metrics
        pt_correct = 0
        onnx_correct = 0
        total_predictions = 0
        pt_inference_times = []
        onnx_inference_times = []
        pt_confidences = []
        onnx_confidences = []
        agreement_count = 0
        
        # Evaluate each image
        for i, (image, target_class) in enumerate(tqdm(validation_data, desc="Comparing models")):
            # PyTorch prediction
            pt_class, pt_conf, pt_time = self.predict_pt_model(image)
            
            # ONNX prediction
            onnx_class, onnx_conf, onnx_time = self.predict_onnx_model(image)
            
            # Update metrics
            if pt_class == target_class:
                pt_correct += 1
            if onnx_class == target_class:
                onnx_correct += 1
            if pt_class == onnx_class:
                agreement_count += 1
            
            total_predictions += 1
            pt_inference_times.append(pt_time)
            onnx_inference_times.append(onnx_time)
            pt_confidences.append(pt_conf)
            onnx_confidences.append(onnx_conf)
            
            # Store results for detailed analysis
            self.pt_results.append({
                'predicted_class': pt_class,
                'confidence': pt_conf,
                'inference_time': pt_time,
                'correct': pt_class == target_class
            })
            
            self.onnx_results.append({
                'predicted_class': onnx_class,
                'confidence': onnx_conf,
                'inference_time': onnx_time,
                'correct': onnx_class == target_class
            })
        
        # Calculate metrics
        pt_accuracy = pt_correct / total_predictions if total_predictions > 0 else 0.0
        onnx_accuracy = onnx_correct / total_predictions if total_predictions > 0 else 0.0
        agreement_rate = agreement_count / total_predictions if total_predictions > 0 else 0.0
        
        pt_avg_time = np.mean(pt_inference_times) if pt_inference_times else 0.0
        onnx_avg_time = np.mean(onnx_inference_times) if onnx_inference_times else 0.0
        
        pt_avg_conf = np.mean(pt_confidences) if pt_confidences else 0.0
        onnx_avg_conf = np.mean(onnx_confidences) if onnx_confidences else 0.0
        
        results = {
            'pt_model': {
                'accuracy': pt_accuracy,
                'avg_inference_time': pt_avg_time,
                'avg_confidence': pt_avg_conf,
                'correct_predictions': pt_correct,
                'total_predictions': total_predictions
            },
            'onnx_model': {
                'accuracy': onnx_accuracy,
                'avg_inference_time': onnx_avg_time,
                'avg_confidence': onnx_avg_conf,
                'correct_predictions': onnx_correct,
                'total_predictions': total_predictions
            },
            'comparison': {
                'agreement_rate': agreement_rate,
                'accuracy_difference': pt_accuracy - onnx_accuracy,
                'speed_ratio': pt_avg_time / onnx_avg_time if onnx_avg_time > 0 else 0.0,
                'confidence_difference': pt_avg_conf - onnx_avg_conf
            }
        }
        
        logger.info(f"Evaluation completed:")
        logger.info(f"  PyTorch Accuracy: {pt_accuracy:.4f}")
        logger.info(f"  ONNX Accuracy: {onnx_accuracy:.4f}")
        logger.info(f"  Agreement Rate: {agreement_rate:.4f}")
        logger.info(f"  Accuracy Difference: {results['comparison']['accuracy_difference']:.4f}")
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_path: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.
        
        Args:
            results: Evaluation results
            output_path: Path to save report
            
        Returns:
            Report dictionary
        """
        report = {
            'model_info': {
                'pt_model_path': self.pt_model_path,
                'onnx_model_path': self.onnx_model_path,
                'num_classes': self.num_classes
            },
            'evaluation_results': results,
            'recommendations': self._generate_recommendations(results)
        }
        
        # Save report
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to: {output_path}")
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        pt_acc = results['pt_model']['accuracy']
        onnx_acc = results['onnx_model']['accuracy']
        agreement = results['comparison']['agreement_rate']
        acc_diff = results['comparison']['accuracy_difference']
        
        # Accuracy-based recommendations
        if abs(acc_diff) < 0.01:
            recommendations.append("Excellent conversion quality - accuracy difference is minimal")
        elif abs(acc_diff) < 0.05:
            recommendations.append("Good conversion quality - small accuracy difference is acceptable")
        else:
            recommendations.append("Significant accuracy difference - consider reviewing conversion process")
        
        # Agreement-based recommendations
        if agreement > 0.95:
            recommendations.append("High model agreement - ONNX model closely matches PyTorch behavior")
        elif agreement > 0.90:
            recommendations.append("Good model agreement - minor differences in predictions")
        else:
            recommendations.append("Low model agreement - significant differences in predictions")
        
        # Performance-based recommendations
        speed_ratio = results['comparison']['speed_ratio']
        if speed_ratio > 1.5:
            recommendations.append("ONNX model is significantly faster - good optimization")
        elif speed_ratio < 0.5:
            recommendations.append("ONNX model is slower - check optimization settings")
        
        # General recommendations
        recommendations.append("Test quantized model on real MaixCam hardware for final validation")
        recommendations.append("Monitor per-class performance differences if accuracy varies significantly")
        
        return recommendations
    
    def _print_summary(self, report: Dict[str, Any]):
        """Print comparison summary."""
        print("\n" + "="*60)
        print("📊 MODEL PERFORMANCE COMPARISON SUMMARY")
        print("="*60)
        
        results = report['evaluation_results']
        pt = results['pt_model']
        onnx = results['onnx_model']
        comp = results['comparison']
        
        print(f"\n🎯 Model Information:")
        print(f"  • Classes: {report['model_info']['num_classes']}")
        print(f"  • PyTorch Model: {Path(report['model_info']['pt_model_path']).name}")
        print(f"  • ONNX Model: {Path(report['model_info']['onnx_model_path']).name}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"  PyTorch Model:")
        print(f"    • Accuracy: {pt['accuracy']:.4f}")
        print(f"    • Avg Inference Time: {pt['avg_inference_time']:.4f}s")
        print(f"    • Avg Confidence: {pt['avg_confidence']:.4f}")
        
        print(f"  ONNX Model:")
        print(f"    • Accuracy: {onnx['accuracy']:.4f}")
        print(f"    • Avg Inference Time: {onnx['avg_inference_time']:.4f}s")
        print(f"    • Avg Confidence: {onnx['avg_confidence']:.4f}")
        
        print(f"\n🔍 Comparison:")
        print(f"  • Agreement Rate: {comp['agreement_rate']:.4f}")
        print(f"  • Accuracy Difference: {comp['accuracy_difference']:.4f}")
        print(f"  • Speed Ratio: {comp['speed_ratio']:.2f}x")
        print(f"  • Confidence Difference: {comp['confidence_difference']:.4f}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX model performance")
    parser.add_argument("--pt-model", required=True, help="Path to best.pt model file")
    parser.add_argument("--onnx-model", required=True, help="Path to ONNX model file")
    parser.add_argument("--classes", required=True, help="Path to classes.txt file")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum samples to evaluate")
    parser.add_argument("--output", help="Path to save comparison report")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.pt_model):
        logger.error(f"PyTorch model file not found: {args.pt_model}")
        return 1
    
    if not os.path.exists(args.onnx_model):
        logger.error(f"ONNX model file not found: {args.onnx_model}")
        return 1
    
    if not os.path.exists(args.classes):
        logger.error(f"Classes file not found: {args.classes}")
        return 1
    
    try:
        # Initialize comparator
        comparator = ModelPerformanceComparator(args.pt_model, args.onnx_model, args.classes)
        
        # Load validation data
        validation_data = comparator.load_validation_data()
        if not validation_data:
            logger.error("Failed to load validation data")
            return 1
        
        # Evaluate models
        results = comparator.evaluate_model(validation_data, args.max_samples)
        
        # Generate report
        report = comparator.generate_report(results, args.output)
        
        logger.info("✅ Model performance comparison completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
