#!/usr/bin/env python3
"""
Data preprocessing script for Pokemon classifier.
Prepares raw images for YOLO training format with multiprocessing.
"""

import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import logging
import json
import multiprocessing as mp
from functools import partial
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_single_image_worker(args):
    """
    Worker function for multiprocessing image processing.
    
    Args:
        args: Tuple of (image_path, pokemon_name, output_dir, image_size)
        
    Returns:
        Tuple of (success, processed_path, pokemon_name, pokemon_id) or (False, None, None, None)
    """
    image_path, pokemon_name, output_dir, image_size = args
    
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return False, None, None, None
        
        # Resize to standard size
        img_resized = cv2.resize(img, (image_size, image_size))
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Save processed image
        output_path = Path(output_dir) / "images" / pokemon_name / f"{image_path.stem}_processed.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_path), (img_normalized * 255).astype(np.uint8))
        
        return True, str(output_path), pokemon_name, image_path.parent.name
        
    except Exception as e:
        return False, None, None, None

# Import Pokemon names from separate file
import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # Try direct import
    from pokemon_names import POKEMON_NAMES
except ImportError:
    try:
        # Try absolute import (when run as script)
        from src.data.pokemon_names import POKEMON_NAMES
    except ImportError:
        try:
            # Try relative import (when run as module)
            from .pokemon_names import POKEMON_NAMES
        except ImportError:
            # Fallback to basic mapping if file doesn't exist
            POKEMON_NAMES = {f"{i:04d}": f"pokemon_{i:04d}" for i in range(1, 1026)}

class PokemonDataPreprocessor:
    """Preprocess Pokemon images for YOLO training with multiprocessing."""
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """Initialize preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.image_size = self.config['processing']['image_size']
        
        # Create output directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "images").mkdir(exist_ok=True)
        (self.processed_dir / "metadata").mkdir(exist_ok=True)
    
    def process_all_pokemon_dataset(self, dataset_path: str, num_workers: int = None) -> Dict:
        """
        Process the full 1025 Pokemon dataset with multiprocessing.
        
        Args:
            dataset_path: Path to the downloaded dataset
            num_workers: Number of worker processes (default: CPU count)
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing dataset from: {dataset_path}")
        start_time = time.time()
        
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        # Set number of workers
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
        logger.info(f"Using {num_workers} worker processes")
        
        # Expected structure:
        # dataset_path/
        #   ├── 0001/  # Bulbasaur
        #   │   ├── 0001bulbasaur-0.jpg
        #   │   ├── 0001Bulbasaur29.jpg
        #   │   └── ...
        #   ├── 0002/  # Ivysaur
        #   │   └── ...
        #   └── ...
        
        pokemon_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()]
        pokemon_dirs.sort(key=lambda x: int(x.name))  # Sort by number
        logger.info(f"Found {len(pokemon_dirs)} Pokemon directories")
        
        # Prepare all image processing tasks
        all_tasks = []
        for pokemon_dir in pokemon_dirs:
            pokemon_id = pokemon_dir.name
            pokemon_name = POKEMON_NAMES.get(pokemon_id, f"unknown_{pokemon_id}")
            
            # Get all images for this Pokemon
            image_files = list(pokemon_dir.glob("*.jpg")) + list(pokemon_dir.glob("*.png"))
            
            for img_file in image_files:
                all_tasks.append((img_file, pokemon_name, self.processed_dir, self.image_size))
        
        logger.info(f"Total images to process: {len(all_tasks)}")
        
        # Process images with multiprocessing
        processed_images = []
        pokemon_names = []
        pokemon_ids = []
        
        with mp.Pool(processes=num_workers) as pool:
            # Use tqdm for progress tracking
            results = list(tqdm(
                pool.imap(process_single_image_worker, all_tasks),
                total=len(all_tasks),
                desc="Processing images"
            ))
        
        # Collect results
        for success, processed_path, pokemon_name, pokemon_id in results:
            if success and processed_path:
                processed_images.append(processed_path)
                pokemon_names.append(pokemon_name)
                pokemon_ids.append(pokemon_id)
        
        # Create metadata
        metadata = {
            'image_paths': processed_images,
            'pokemon_names': pokemon_names,
            'pokemon_ids': pokemon_ids,
            'total_images': len(processed_images),
            'unique_pokemon': len(set(pokemon_names)),
            'pokemon_mapping': POKEMON_NAMES,
            'processing_time': time.time() - start_time
        }
        
        # Save metadata
        metadata_path = self.processed_dir / "metadata" / "dataset_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save Pokemon mapping
        mapping_path = self.processed_dir / "metadata" / "pokemon_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(POKEMON_NAMES, f, indent=2)
        
        logger.info(f"Processed {len(processed_images)} images from {len(set(pokemon_names))} Pokemon")
        logger.info(f"Processing time: {metadata['processing_time']:.2f} seconds")
        return metadata
    
    def create_yolo_dataset(self, output_dir: str = "data/processed/yolo_dataset"):
        """
        Create YOLO-compatible dataset structure.
        
        Args:
            output_dir: Output directory for YOLO dataset
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create YOLO dataset structure
        (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "test").mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        metadata_path = self.processed_dir / "metadata" / "dataset_info.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create Pokemon name to class ID mapping
        unique_pokemon = list(set(metadata['pokemon_names']))
        unique_pokemon.sort()  # Sort for consistent ordering
        pokemon_to_id = {name: idx for idx, name in enumerate(unique_pokemon)}
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        # Group by Pokemon to ensure balanced splits
        pokemon_groups = {}
        for img_path, pokemon_name in zip(metadata['image_paths'], metadata['pokemon_names']):
            if pokemon_name not in pokemon_groups:
                pokemon_groups[pokemon_name] = []
            pokemon_groups[pokemon_name].append(img_path)
        
        train_images, temp_images = train_test_split(
            list(pokemon_groups.items()), 
            test_size=0.3, 
            random_state=42
        )
        
        val_images, test_images = train_test_split(
            temp_images, 
            test_size=0.5, 
            random_state=42
        )
        
        # Process splits
        self._process_split(train_images, output_dir, "train", pokemon_to_id)
        self._process_split(val_images, output_dir, "val", pokemon_to_id)
        self._process_split(test_images, output_dir, "test", pokemon_to_id)
        
        # Save class mapping
        class_mapping = {idx: name for name, idx in pokemon_to_id.items()}
        with open(output_dir / "classes.txt", 'w') as f:
            for idx in range(len(class_mapping)):
                f.write(f"{class_mapping[idx]}\n")
        
        logger.info(f"Created YOLO dataset at {output_dir}")
        logger.info(f"Classes: {len(class_mapping)}")
    
    def _process_split(self, split_data, output_dir, split_name, pokemon_to_id):
        """Process a single data split."""
        for pokemon_name, image_paths in split_data:
            class_id = pokemon_to_id[pokemon_name]
            
            for img_path in image_paths:
                # Copy image
                src_path = Path(img_path)
                dst_path = output_dir / "images" / split_name / src_path.name
                
                import shutil
                shutil.copy2(src_path, dst_path)
                
                # Create YOLO label file (classification format)
                label_path = output_dir / "labels" / split_name / f"{src_path.stem}.txt"
                with open(label_path, 'w') as f:
                    f.write(f"{class_id}\n")

def main():
    parser = argparse.ArgumentParser(description="Preprocess Pokemon dataset with multiprocessing")
    parser.add_argument("--dataset_path", required=True, help="Path to raw dataset")
    parser.add_argument("--config", default="configs/data_config.yaml", help="Config file path")
    parser.add_argument("--create_yolo_dataset", action="store_true", help="Create YOLO dataset structure")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = PokemonDataPreprocessor(args.config)
    
    # Process dataset with multiprocessing
    metadata = preprocessor.process_all_pokemon_dataset(args.dataset_path, args.num_workers)
    
    # Create YOLO dataset if requested
    if args.create_yolo_dataset:
        preprocessor.create_yolo_dataset()
    
    print(f"Processing complete! Processed {metadata['total_images']} images from {metadata['unique_pokemon']} Pokemon")
    print(f"Processing time: {metadata['processing_time']:.2f} seconds")

if __name__ == "__main__":
    main() 