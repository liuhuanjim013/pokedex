#!/usr/bin/env python3
"""
YOLO Model Evaluator
Handles model evaluation and performance metrics calculation
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOEvaluator:
    """Evaluates YOLO model performance and generates reports."""
    
    def __init__(self, model: YOLO, config: Dict[str, Any]):
        """Initialize evaluator with model and configuration."""
        self.model = model
        self.config = config
        self.results = {}
        
        logger.info("YOLOEvaluator initialized")
    
    def evaluate_model(self, test_data: str) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Path to test data or dataset name
            
        Returns:
            Evaluation metrics dictionary
        """
        logger.info("Starting model evaluation...")
        
        try:
            # Run validation
            results = self.model.val(data=test_data)
            
            # Extract metrics
            metrics = {
                'mAP50': results.get('metrics/mAP50(B)', 0),
                'mAP50-95': results.get('metrics/mAP50-95(B)', 0),
                'precision': results.get('metrics/precision(B)', 0),
                'recall': results.get('metrics/recall(B)', 0),
                'accuracy': self._calculate_accuracy(results),
                'top5_accuracy': self._calculate_top5_accuracy(results),
            }
            
            self.results = metrics
            logger.info(f"Evaluation completed: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}
    
    def _calculate_accuracy(self, results: Dict[str, Any]) -> float:
        """Calculate classification accuracy."""
        try:
            # Extract predictions and targets
            predictions = results.get('predictions', [])
            targets = results.get('targets', [])
            
            if not predictions or not targets:
                return 0.0
            
            correct = 0
            total = 0
            
            for pred, target in zip(predictions, targets):
                if pred['class'] == target['class']:
                    correct += 1
                total += 1
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate accuracy: {e}")
            return 0.0
    
    def _calculate_top5_accuracy(self, results: Dict[str, Any]) -> float:
        """Calculate top-5 classification accuracy."""
        try:
            # This would need to be implemented based on model output format
            # For now, return a placeholder
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate top-5 accuracy: {e}")
            return 0.0
    
    def generate_confusion_matrix(self, test_data: str, save_path: str = None) -> np.ndarray:
        """
        Generate confusion matrix for classification results.
        
        Args:
            test_data: Path to test data
            save_path: Path to save confusion matrix plot
            
        Returns:
            Confusion matrix as numpy array
        """
        try:
            # This would need to be implemented based on model predictions
            # For now, return a placeholder
            logger.info("Confusion matrix generation not yet implemented")
            return np.zeros((1025, 1025))
            
        except Exception as e:
            logger.error(f"Failed to generate confusion matrix: {e}")
            return np.zeros((1025, 1025))
    
    def create_evaluation_report(self, save_path: str = None) -> str:
        """
        Create comprehensive evaluation report.
        
        Args:
            save_path: Path to save report
            
        Returns:
            Report content as string
        """
        if not self.results:
            logger.warning("No evaluation results available")
            return ""
        
        report = f"""
# YOLO Model Evaluation Report

## Model Configuration
- Model: {self.config.get('model', {}).get('name', 'yolov3')}
- Classes: {self.config.get('model', {}).get('classes', 1025)}
- Image Size: {self.config.get('model', {}).get('img_size', 416)}

## Performance Metrics
- mAP50: {self.results.get('mAP50', 0):.4f}
- mAP50-95: {self.results.get('mAP50-95', 0):.4f}
- Precision: {self.results.get('precision', 0):.4f}
- Recall: {self.results.get('recall', 0):.4f}
- Accuracy: {self.results.get('accuracy', 0):.4f}
- Top-5 Accuracy: {self.results.get('top5_accuracy', 0):.4f}

## Training Configuration
- Epochs: {self.config.get('training', {}).get('epochs', 100)}
- Batch Size: {self.config.get('training', {}).get('batch_size', 16)}
- Learning Rate: {self.config.get('training', {}).get('learning_rate', 0.001)}

## Dataset Information
- Dataset: {self.config.get('data', {}).get('dataset', 'pokemon-yolo-1025')}
- Classes: {self.config.get('model', {}).get('classes', 1025)}

## Recommendations
- Model performance analysis
- Potential improvements
- Deployment considerations
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to: {save_path}")
        
        return report
    
    def compare_models(self, baseline_results: Dict[str, Any], improved_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare baseline and improved model results.
        
        Args:
            baseline_results: Results from baseline model
            improved_results: Results from improved model
            
        Returns:
            Comparison metrics
        """
        comparison = {
            'mAP50_improvement': improved_results.get('mAP50', 0) - baseline_results.get('mAP50', 0),
            'mAP50-95_improvement': improved_results.get('mAP50-95', 0) - baseline_results.get('mAP50-95', 0),
            'precision_improvement': improved_results.get('precision', 0) - baseline_results.get('precision', 0),
            'recall_improvement': improved_results.get('recall', 0) - baseline_results.get('recall', 0),
            'accuracy_improvement': improved_results.get('accuracy', 0) - baseline_results.get('accuracy', 0),
        }
        
        logger.info(f"Model comparison: {comparison}")
        return comparison
    
    def plot_training_curves(self, wandb_run_ids: List[str], save_path: str = None):
        """
        Plot training curves from W&B runs.
        
        Args:
            wandb_run_ids: List of W&B run IDs to plot
            save_path: Path to save plot
        """
        try:
            import wandb
            
            # This would need to be implemented to fetch data from W&B
            logger.info("Training curve plotting not yet implemented")
            
        except Exception as e:
            logger.error(f"Failed to plot training curves: {e}")
    
    def export_results(self, export_path: str):
        """
        Export evaluation results to file.
        
        Args:
            export_path: Path to export results
        """
        try:
            import json
            
            with open(export_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"Results exported to: {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}") 