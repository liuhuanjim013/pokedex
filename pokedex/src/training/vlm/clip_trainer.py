#!/usr/bin/env python3
"""
CLIP training script for Pokemon classifier.
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any
import logging
import wandb
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTrainer, CLIPTrainingArguments
from datasets import load_dataset
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIPPokemonTrainer:
    """Train CLIP model for Pokemon classification."""
    
    def __init__(self, config_path: str = "configs/clip/training_config.yaml"):
        """Initialize trainer with CLIP-specific configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize W&B
        self._setup_wandb()
        
        # Model settings
        self.model_name = self.config['model']['name']
        self.num_classes = self.config['model']['classes']
        self.img_size = self.config['model']['img_size']
        
        # Training settings
        self.epochs = self.config['training']['epochs']
        self.batch_size = self.config['training']['batch_size']
        self.learning_rate = self.config['training']['learning_rate']
        
        # Hardware settings
        self.device = self.config['hardware']['device']
        
        logger.info(f"Initialized CLIP trainer for {self.num_classes} classes")
    
    def _setup_wandb(self):
        """Set up Weights & Biases tracking."""
        wandb_config = self.config['wandb']
        
        wandb.init(
            project=wandb_config['project'],
            name=wandb_config['name'],
            tags=wandb_config['tags'],
            config={
                'model': self.config['model'],
                'training': self.config['training'],
                'data': self.config['data'],
                'clip': self.config['clip']
            }
        )
        
        logger.info(f"W&B initialized: {wandb.run.name}")
    
    def load_model_and_processor(self):
        """Load CLIP model and processor."""
        clip_config = self.config['clip']
        
        # Load model and processor
        model = CLIPModel.from_pretrained(clip_config['model_name'])
        processor = CLIPProcessor.from_pretrained(clip_config['model_name'])
        
        # Configure for classification
        model.config.num_labels = self.num_classes
        
        return model, processor
    
    def prepare_dataset(self, dataset_path: str):
        """Prepare dataset for CLIP training."""
        # Load dataset from Hugging Face or local path
        if dataset_path.startswith("your-username/"):
            dataset = load_dataset(dataset_path)
        else:
            dataset = load_dataset("imagefolder", data_dir=dataset_path)
        
        # Create text prompts for each Pokemon
        clip_config = self.config['clip']
        text_prompts = clip_config['text_prompts']
        
        def create_text_prompts(example):
            pokemon_name = example['pokemon_name']
            texts = [prompt.format(pokemon_name=pokemon_name) for prompt in text_prompts]
            example['text'] = texts
            return example
        
        # Apply text prompts
        dataset = dataset.map(create_text_prompts, batched=False)
        
        return dataset
    
    def train(self, dataset_path: str, model_save_dir: str = "models/checkpoints/clip"):
        """
        Train CLIP model for Pokemon classification.
        
        Args:
            dataset_path: Path to dataset (Hugging Face or local)
            model_save_dir: Directory to save trained models
        """
        logger.info("Starting CLIP training for Pokemon classification")
        
        # Create model save directory
        model_save_dir = Path(model_save_dir)
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and processor
        model, processor = self.load_model_and_processor()
        
        # Prepare dataset
        dataset = self.prepare_dataset(dataset_path)
        
        # CLIP-specific training arguments
        training_args = CLIPTrainingArguments(
            output_dir=str(model_save_dir),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.config['training']['weight_decay'],
            warmup_steps=self.config['training']['scheduler']['warmup_epochs'] * len(dataset['train']) // self.batch_size,
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            push_to_hub=False,
            report_to="wandb",
            dataloader_num_workers=self.config['hardware']['num_workers'],
            dataloader_pin_memory=self.config['hardware']['pin_memory'],
        )
        
        # Create trainer
        trainer = CLIPTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=processor.tokenizer,
            data_collator=self._collate_fn,
        )
        
        # Start training
        logger.info(f"Training CLIP with {self.epochs} epochs, batch size {self.batch_size}")
        results = trainer.train()
        
        # Save model
        trainer.save_model(str(model_save_dir / "final"))
        processor.save_pretrained(str(model_save_dir / "final"))
        
        # Log final metrics
        self._log_final_metrics(results)
        
        wandb.finish()
        return results
    
    def _collate_fn(self, batch):
        """Custom collate function for CLIP training."""
        images = [item['image'] for item in batch]
        texts = [item['text'] for item in batch]
        
        # Process images and texts
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config['clip']['max_text_length']
        )
        
        return inputs
    
    def _log_final_metrics(self, results):
        """Log final training metrics to W&B."""
        wandb.log({
            'final/train_loss': results.training_loss,
            'final/eval_loss': results.metrics.get('eval_loss', 0),
            'final/learning_rate': results.metrics.get('learning_rate', 0),
        })
    
    def evaluate(self, model_path: str, dataset_path: str):
        """
        Evaluate trained CLIP model.
        
        Args:
            model_path: Path to trained model
            dataset_path: Path to dataset
        """
        logger.info(f"Evaluating CLIP model: {model_path}")
        
        # Load model and processor
        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)
        
        # Load dataset
        dataset = self.prepare_dataset(dataset_path)
        
        # Evaluation logic here
        # This would involve running inference on test set
        # and computing accuracy metrics
        
        return {"accuracy": 0.0}  # Placeholder

def main():
    parser = argparse.ArgumentParser(description="Train CLIP model for Pokemon classification")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset (Hugging Face or local)")
    parser.add_argument("--config", default="configs/clip/training_config.yaml", help="Config file path")
    parser.add_argument("--model_save_dir", default="models/checkpoints/clip", help="Model save directory")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model after training")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CLIPPokemonTrainer(args.config)
    
    # Train model
    results = trainer.train(args.dataset_path, args.model_save_dir)
    
    # Evaluate if requested
    if args.evaluate:
        model_path = Path(args.model_save_dir) / "final"
        if model_path.exists():
            trainer.evaluate(str(model_path), args.dataset_path)
    
    print("CLIP training completed!")

if __name__ == "__main__":
    main() 