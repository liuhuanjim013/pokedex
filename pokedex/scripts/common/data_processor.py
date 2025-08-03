#!/usr/bin/env python3
"""
Shared data processing pipeline for Pokemon classifier.
Processes raw data once and creates model-specific formats.
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging
import json
import shutil
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SharedDataProcessor:
    """Process raw data once and create model-specific formats."""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """Initialize processor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        
        # Create output directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "images").mkdir(exist_ok=True)
        (self.processed_dir / "metadata").mkdir(exist_ok=True)
    
    def process_raw_data(self, dataset_path: str) -> Dict:
        """
        Process raw data once for all models.
        
        Args:
            dataset_path: Path to raw dataset
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing raw data from: {dataset_path}")
        
        # Use the existing preprocessing script
        from src.data.preprocessing import PokemonDataPreprocessor
        
        preprocessor = PokemonDataPreprocessor("configs/data_config.yaml")
        metadata = preprocessor.process_gen1_3_dataset(dataset_path)
        
        logger.info(f"Raw data processing completed: {metadata['total_images']} images")
        return metadata
    
    def create_yolo_dataset(self, output_dir: str = "data/processed/yolo_dataset"):
        """Create YOLO format dataset."""
        logger.info("Creating YOLO format dataset...")
        
        from src.data.preprocessing import PokemonDataPreprocessor
        preprocessor = PokemonDataPreprocessor("configs/data_config.yaml")
        preprocessor.create_yolo_dataset(output_dir)
        
        logger.info(f"YOLO dataset created at {output_dir}")
    
    def create_clip_dataset(self, output_dir: str = "data/processed/clip_dataset"):
        """Create CLIP format dataset."""
        logger.info("Creating CLIP format dataset...")
        
        # Load processed images and metadata
        processed_dir = Path("data/processed")
        metadata_path = processed_dir / "metadata" / "dataset_info.json"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create CLIP dataset structure
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images and create CLIP-specific metadata
        for img_path, pokemon_name in zip(metadata['image_paths'], metadata['pokemon_names']):
            src_path = Path(img_path)
            dst_path = output_dir / f"{pokemon_name}_{src_path.name}"
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
        
        # Create CLIP metadata
        clip_metadata = {
            'total_images': len(metadata['image_paths']),
            'unique_pokemon': len(set(metadata['pokemon_names'])),
            'pokemon_names': list(set(metadata['pokemon_names'])),
            'format': 'clip'
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(clip_metadata, f, indent=2)
        
        logger.info(f"CLIP dataset created at {output_dir}")
    
    def create_smolvm_dataset(self, output_dir: str = "data/processed/smolvm_dataset"):
        """Create SMoLVM format dataset."""
        logger.info("Creating SMoLVM format dataset...")
        
        # Similar to CLIP but with SMoLVM-specific formatting
        # For now, use same structure as CLIP
        self.create_clip_dataset(output_dir)
        
        # Update metadata
        metadata_path = Path(output_dir) / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['format'] = 'smolvm'
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"SMoLVM dataset created at {output_dir}")
    
    def upload_to_huggingface(self, dataset_path: str, dataset_name: str, token: str = None):
        """Upload dataset to Hugging Face."""
        logger.info(f"Uploading {dataset_path} to Hugging Face as {dataset_name}")
        
        from scripts.upload_dataset import PokemonDatasetUploader
        
        uploader = PokemonDatasetUploader(token)
        
        if Path(dataset_path).name == "yolo_dataset":
            uploader.create_yolo_dataset(dataset_path, dataset_name)
        else:
            uploader.create_dataset_from_processed(dataset_path, dataset_name)
    
    def process_all_formats(self, dataset_path: str, upload_to_hf: bool = False, 
                          hf_token: str = None):
        """
        Process raw data and create all model-specific formats.
        
        Args:
            dataset_path: Path to raw dataset
            upload_to_hf: Whether to upload to Hugging Face
            hf_token: Hugging Face token
        """
        logger.info("Starting complete data processing pipeline...")
        
        # Step 1: Process raw data once
        metadata = self.process_raw_data(dataset_path)
        
        # Step 2: Create model-specific formats
        self.create_yolo_dataset()
        self.create_clip_dataset()
        self.create_smolvm_dataset()
        
        # Step 3: Upload to Hugging Face if requested
        if upload_to_hf and hf_token:
            self.upload_to_huggingface("data/processed/yolo_dataset", 
                                     "your-username/pokemon-yolo-gen1-3", hf_token)
            self.upload_to_huggingface("data/processed/clip_dataset", 
                                     "your-username/pokemon-clip-gen1-3", hf_token)
            self.upload_to_huggingface("data/processed/smolvm_dataset", 
                                     "your-username/pokemon-smolvm-gen1-3", hf_token)
        
        logger.info("Complete data processing pipeline finished!")
        return metadata

def main():
    parser = argparse.ArgumentParser(description="Process Pokemon dataset for all models")
    parser.add_argument("--dataset_path", required=True, help="Path to raw dataset")
    parser.add_argument("--config", default="configs/data_config.yaml", help="Config file path")
    parser.add_argument("--upload_to_hf", action="store_true", help="Upload to Hugging Face")
    parser.add_argument("--hf_token", help="Hugging Face token")
    parser.add_argument("--format", choices=["all", "yolo", "clip", "smolvm"], 
                       default="all", help="Which format to create")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = SharedDataProcessor(args.config)
    
    if args.format == "all":
        # Process everything
        processor.process_all_formats(args.dataset_path, args.upload_to_hf, args.hf_token)
    else:
        # Process raw data first
        processor.process_raw_data(args.dataset_path)
        
        # Create specific format
        if args.format == "yolo":
            processor.create_yolo_dataset()
        elif args.format == "clip":
            processor.create_clip_dataset()
        elif args.format == "smolvm":
            processor.create_smolvm_dataset()
    
    print("Data processing completed!")

if __name__ == "__main__":
    main() 