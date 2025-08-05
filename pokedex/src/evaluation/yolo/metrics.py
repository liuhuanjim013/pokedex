#!/usr/bin/env python3
"""
YOLO Performance Metrics
Calculation functions for YOLO model evaluation
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

def calculate_map(predictions: List[Dict], targets: List[Dict], iou_threshold: float = 0.5) -> float:
    """
    Calculate mean Average Precision (mAP).
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_threshold: IoU threshold for positive detection
        
    Returns:
        mAP value
    """
    try:
        # This is a simplified implementation
        # Real mAP calculation would be more complex
        
        if not predictions or not targets:
            return 0.0
        
        # Calculate IoU for each prediction-target pair
        ious = []
        for pred in predictions:
            for target in targets:
                iou = calculate_iou(pred['bbox'], target['bbox'])
                ious.append(iou)
        
        # Count true positives (IoU > threshold)
        true_positives = sum(1 for iou in ious if iou > iou_threshold)
        
        # Calculate precision
        precision = true_positives / len(predictions) if predictions else 0.0
        
        return precision
        
    except Exception as e:
        logger.error(f"Failed to calculate mAP: {e}")
        return 0.0

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU value
    """
    try:
        # Convert to [x1, y1, x2, y2] format if needed
        if len(box1) == 4 and len(box2) == 4:
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
        else:
            return 0.0
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Failed to calculate IoU: {e}")
        return 0.0

def calculate_accuracy(predictions: List[int], targets: List[int]) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        predictions: List of predicted class indices
        targets: List of target class indices
        
    Returns:
        Accuracy value
    """
    try:
        if not predictions or not targets:
            return 0.0
        
        if len(predictions) != len(targets):
            logger.warning("Predictions and targets have different lengths")
            return 0.0
        
        correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
        accuracy = correct / len(predictions)
        
        return accuracy
        
    except Exception as e:
        logger.error(f"Failed to calculate accuracy: {e}")
        return 0.0

def calculate_top5_accuracy(predictions: List[List[int]], targets: List[int]) -> float:
    """
    Calculate top-5 classification accuracy.
    
    Args:
        predictions: List of top-5 predicted class indices
        targets: List of target class indices
        
    Returns:
        Top-5 accuracy value
    """
    try:
        if not predictions or not targets:
            return 0.0
        
        if len(predictions) != len(targets):
            logger.warning("Predictions and targets have different lengths")
            return 0.0
        
        correct = 0
        for pred, target in zip(predictions, targets):
            if target in pred[:5]:  # Check if target is in top-5
                correct += 1
        
        accuracy = correct / len(predictions)
        return accuracy
        
    except Exception as e:
        logger.error(f"Failed to calculate top-5 accuracy: {e}")
        return 0.0

def calculate_precision_recall(predictions: List[Dict], targets: List[Dict], iou_threshold: float = 0.5) -> Tuple[float, float]:
    """
    Calculate precision and recall.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_threshold: IoU threshold for positive detection
        
    Returns:
        Tuple of (precision, recall)
    """
    try:
        if not predictions or not targets:
            return 0.0, 0.0
        
        # Count true positives, false positives, false negatives
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # For each prediction, check if it matches any target
        for pred in predictions:
            matched = False
            for target in targets:
                iou = calculate_iou(pred['bbox'], target['bbox'])
                if iou > iou_threshold:
                    true_positives += 1
                    matched = True
                    break
            if not matched:
                false_positives += 1
        
        # False negatives = targets not matched by any prediction
        for target in targets:
            matched = False
            for pred in predictions:
                iou = calculate_iou(pred['bbox'], target['bbox'])
                if iou > iou_threshold:
                    matched = True
                    break
            if not matched:
                false_negatives += 1
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        return precision, recall
        
    except Exception as e:
        logger.error(f"Failed to calculate precision/recall: {e}")
        return 0.0, 0.0

def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall.
    
    Args:
        precision: Precision value
        recall: Recall value
        
    Returns:
        F1 score
    """
    try:
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
        
    except Exception as e:
        logger.error(f"Failed to calculate F1 score: {e}")
        return 0.0

def calculate_class_accuracy(predictions: List[int], targets: List[int], num_classes: int) -> Dict[int, float]:
    """
    Calculate per-class accuracy.
    
    Args:
        predictions: List of predicted class indices
        targets: List of target class indices
        num_classes: Number of classes
        
    Returns:
        Dictionary mapping class index to accuracy
    """
    try:
        class_correct = {i: 0 for i in range(num_classes)}
        class_total = {i: 0 for i in range(num_classes)}
        
        for pred, target in zip(predictions, targets):
            class_total[target] += 1
            if pred == target:
                class_correct[target] += 1
        
        class_accuracy = {}
        for class_idx in range(num_classes):
            if class_total[class_idx] > 0:
                class_accuracy[class_idx] = class_correct[class_idx] / class_total[class_idx]
            else:
                class_accuracy[class_idx] = 0.0
        
        return class_accuracy
        
    except Exception as e:
        logger.error(f"Failed to calculate class accuracy: {e}")
        return {}

def calculate_confusion_matrix(predictions: List[int], targets: List[int], num_classes: int) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        predictions: List of predicted class indices
        targets: List of target class indices
        num_classes: Number of classes
        
    Returns:
        Confusion matrix as numpy array
    """
    try:
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
        
        for pred, target in zip(predictions, targets):
            confusion_matrix[target][pred] += 1
        
        return confusion_matrix
        
    except Exception as e:
        logger.error(f"Failed to calculate confusion matrix: {e}")
        return np.zeros((num_classes, num_classes), dtype=np.int32) 