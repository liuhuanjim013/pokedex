#!/usr/bin/env python3
"""
Upload processed Pokemon dataset to Hugging Face Hub.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
import logging
from datasets import Dataset, DatasetDict, Image, Features, Value, ClassLabel
from huggingface_hub import HfApi, login
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PokemonDatasetUploader:
    """Upload Pokemon dataset to Hugging Face Hub."""
    
    def __init__(self, token: str = None):
        """Initialize uploader with Hugging Face token."""
        if token:
            login(token)
        else:
            # Try to use token from environment
            login()
        
        self.api = HfApi()
    
    def create_dataset_from_processed(self, processed_dir: str, dataset_name: str) -> str:
        """
        Create Hugging Face dataset from processed data.
        
        Args:
            processed_dir: Path to processed data directory
            dataset_name: Name for the dataset on Hugging Face Hub
            
        Returns:
            Dataset ID on Hugging Face Hub
        """
        processed_dir = Path(processed_dir)
        
        # Load metadata
        metadata_path = processed_dir / "metadata" / "dataset_info.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Creating dataset from {metadata['total_images']} images")
        
        # Create dataset structure
        dataset_dict = {
            'image': [],
            'pokemon_name': [],
            'label': []
        }
        
        # Create Pokemon name to label mapping
        unique_pokemon = list(set(metadata['pokemon_names']))
        pokemon_to_label = {name: idx for idx, name in enumerate(unique_pokemon)}
        
        # Load images and create dataset
        for img_path, pokemon_name in zip(metadata['image_paths'], metadata['pokemon_names']):
            img_path = Path(img_path)
            
            if img_path.exists():
                dataset_dict['image'].append(str(img_path))
                dataset_dict['pokemon_name'].append(pokemon_name)
                dataset_dict['label'].append(pokemon_to_label[pokemon_name])
            else:
                logger.warning(f"Image not found: {img_path}")
        
        # Create features
        features = Features({
            'image': Image(),
            'pokemon_name': Value('string'),
            'label': ClassLabel(num_classes=len(unique_pokemon), names=unique_pokemon)
        })
        
        # Create dataset
        dataset = Dataset.from_dict(dataset_dict, features=features)
        
        # Add dataset info
        dataset.info.description = f"Pokemon classification dataset with {len(unique_pokemon)} Pokemon species"
        dataset.info.homepage = "https://github.com/your-username/pokemon-classifier"
        dataset.info.license = "MIT"
        
        # Upload to Hugging Face Hub
        logger.info(f"Uploading dataset to {dataset_name}")
        dataset.push_to_hub(dataset_name, private=False)
        
        logger.info(f"Dataset uploaded successfully: https://huggingface.co/datasets/{dataset_name}")
        return dataset_name
    
    def create_yolo_dataset(self, yolo_dir: str, dataset_name: str) -> str:
        """
        Create YOLO-compatible dataset on Hugging Face.
        
        Args:
            yolo_dir: Path to YOLO dataset directory
            dataset_name: Name for the dataset on Hugging Face Hub
            
        Returns:
            Dataset ID on Hugging Face Hub
        """
        yolo_dir = Path(yolo_dir)
        
        # Load class names
        classes_file = yolo_dir / "classes.txt"
        if not classes_file.exists():
            raise FileNotFoundError(f"Classes file not found at {classes_file}")
        
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        logger.info(f"Creating YOLO dataset with {len(class_names)} classes")
        
        # Create dataset for each split
        splits = {}
        
        for split in ['train', 'val', 'test']:
            images_dir = yolo_dir / "images" / split
            labels_dir = yolo_dir / "labels" / split
            
            if not images_dir.exists():
                logger.warning(f"Split {split} not found, skipping")
                continue
            
            # Collect data for this split
            split_data = {
                'image': [],
                'label': [],
                'pokemon_name': []
            }
            
            # Process each image
            for img_file in images_dir.glob("*.jpg"):
                label_file = labels_dir / f"{img_file.stem}.txt"
                
                if label_file.exists():
                    # Read label (class ID)
                    with open(label_file, 'r') as f:
                        class_id = int(f.read().strip())
                    
                    split_data['image'].append(str(img_file))
                    split_data['label'].append(class_id)
                    split_data['pokemon_name'].append(class_names[class_id])
            
            if split_data['image']:
                # Create features
                features = Features({
                    'image': Image(),
                    'label': ClassLabel(num_classes=len(class_names), names=class_names),
                    'pokemon_name': Value('string')
                })
                
                # Create dataset for this split
                splits[split] = Dataset.from_dict(split_data, features=features)
        
        # Create DatasetDict
        dataset_dict = DatasetDict(splits)
        
        # Add dataset info
        dataset_dict.info.description = f"YOLO-compatible Pokemon classification dataset with {len(class_names)} Pokemon species"
        dataset_dict.info.homepage = "https://github.com/your-username/pokemon-classifier"
        dataset_dict.info.license = "MIT"
        
        # Upload to Hugging Face Hub
        logger.info(f"Uploading YOLO dataset to {dataset_name}")
        dataset_dict.push_to_hub(dataset_name, private=False)
        
        logger.info(f"YOLO dataset uploaded successfully: https://huggingface.co/datasets/{dataset_name}")
        return dataset_name

def main():
    parser = argparse.ArgumentParser(description="Upload Pokemon dataset to Hugging Face")
    parser.add_argument("--processed_dir", required=True, help="Path to processed data directory")
    parser.add_argument("--dataset_name", required=True, help="Dataset name on Hugging Face Hub")
    parser.add_argument("--token", help="Hugging Face token")
    parser.add_argument("--yolo_format", action="store_true", help="Upload in YOLO format")
    
    args = parser.parse_args()
    
    # Initialize uploader
    uploader = PokemonDatasetUploader(args.token)
    
    if args.yolo_format:
        # Upload YOLO dataset
        uploader.create_yolo_dataset(args.processed_dir, args.dataset_name)
    else:
        # Upload regular dataset
        uploader.create_dataset_from_processed(args.processed_dir, args.dataset_name)

if __name__ == "__main__":
    main() 