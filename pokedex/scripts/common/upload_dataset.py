#!/usr/bin/env python3
"""
Upload Pokemon dataset to Hugging Face Hub.
Handles proper split organization and dataset card creation.
"""

import os
import sys
import json
import time
import logging
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import partial
from PIL import Image as PILImage
try:
    import backoff
except ImportError:
    logging.error("The 'backoff' package is required. Please install it with:")
    logging.error("  pip install backoff")
    sys.exit(1)
from datasets import Dataset, DatasetDict, Features, Image, ClassLabel, Value, concatenate_datasets
from huggingface_hub import HfApi
from requests.exceptions import RequestException, ConnectionError
from urllib3.exceptions import NameResolutionError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retry_on_network_error(e):
    """Return True if we should retry on this error."""
    if isinstance(e, (ConnectionError, NameResolutionError, RequestException)):
        logging.warning(f"Network error occurred: {str(e)}")
        logging.info("Retrying in a moment...")
        return True
    return False

class PokemonDatasetUploader:
    """Upload Pokemon dataset to Hugging Face Hub with retry mechanism."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize uploader with optional token."""
        self.token = token or os.environ.get("HUGGINGFACE_TOKEN")
        if not self.token:
            raise ValueError("Hugging Face token required")
        
        self.api = HfApi(token=self.token)
    
    @staticmethod
    def process_image_batch(args: Tuple[List[Path], List[Path], Path, Dict, Dict]) -> Tuple[Dict[str, List], int]:
        """Process a batch of images in parallel.
        
        Args:
            args: Tuple of (image_files, label_files, dataset_path, dest_to_raw, dest_to_info)
            
        Returns:
            Tuple of (batch_data, processed_count)
        """
        image_files, label_files, dataset_path, dest_to_raw, dest_to_info = args
        
        batch_data = {
            "image": [],
            "label_file": [],
            "label": [],
            "raw_image": [],
            "pokemon_id": []
        }
        
        processed = 0
        for img_file, lbl_file in zip(image_files, label_files):
            # Verify file pair
            img_stem = img_file.stem
            lbl_stem = lbl_file.stem
            if img_stem != lbl_stem:
                raise ValueError(f"Mismatched files: {img_file} and {lbl_file}")
            
            # Parse Pokemon ID from filename (NNNN_XXX format)
            try:
                pokemon_id = int(img_stem.split('_')[0])
            except (ValueError, IndexError):
                raise ValueError(f"Invalid filename format: {img_file}")
            
            # Load and verify image
            try:
                img = PILImage.open(img_file)
                img.verify()  # Verify it's a valid image
            except Exception as e:
                raise ValueError(f"Invalid image {img_file}: {e}")
            
            # Load and verify label
            with open(lbl_file, "r") as f:
                label_content = f.read().strip()
                try:
                    class_id, x, y, w, h = map(float, label_content.split())
                    if not (class_id.is_integer() and 0 <= class_id < 1025):
                        raise ValueError(f"Invalid class ID in {lbl_file}: {class_id}")
                    if not (x == 0.5 and y == 0.5 and w == 1.0 and h == 1.0):
                        raise ValueError(f"Invalid box format in {lbl_file}")
                except Exception as e:
                    raise ValueError(f"Invalid label format in {lbl_file}: {e}")
            
            # Get raw source from lookup table
            processed_path = str(img_file.resolve())
            if processed_path not in dest_to_raw:
                # Try relative path
                processed_rel = str(img_file.relative_to(dataset_path))
                processed_path = str((dataset_path / processed_rel).resolve())
                if processed_path not in dest_to_raw:
                    raise ValueError(f"Missing raw mapping for {processed_path}")
            
            raw_source = dest_to_raw[processed_path]
            
            # Add to batch
            batch_data["image"].append(str(img_file))
            batch_data["label_file"].append(str(lbl_file))
            batch_data["label"].append(int(class_id))
            batch_data["raw_image"].append(raw_source)
            batch_data["pokemon_id"].append(pokemon_id)
            processed += 1
            
        return batch_data, processed

    def create_yolo_dataset(self, dataset_path: str, dataset_name: str):
        """
        Create and upload YOLO format dataset with proper splits.
        
        Args:
            dataset_path: Path to YOLO format dataset
            dataset_name: Name for Hugging Face dataset (e.g., 'username/dataset-name')
        """
        logging.info(f"Creating YOLO dataset from {dataset_path}")
        
        # Handle relative paths correctly
        if not Path(dataset_path).is_absolute():
            # If relative path, make it relative to the repository root
            repo_root = Path(__file__).resolve().parents[3]  # Go up 3 levels from scripts/common/
            dataset_path = repo_root / dataset_path
        else:
            dataset_path = Path(dataset_path)
            
        dataset_path = dataset_path.resolve()
        logging.info(f"Using absolute path: {dataset_path}")
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
            
        # Generate split statistics
        split_stats = {
            'train': {'images': 0, 'pokemon': set()},
            'validation': {'images': 0, 'pokemon': set()},
            'test': {'images': 0, 'pokemon': set()}
        }
        
        # Load split mapping for traceability
        mapping_file = dataset_path / "split_mapping.json"
        if not mapping_file.exists():
            raise FileNotFoundError(
                f"Split mapping file not found: {mapping_file}\n"
                "Please run prepare_yolo_dataset.py first to generate the dataset."
            )
            
        with open(mapping_file, "r") as f:
            split_mapping = json.load(f)
            
        # Create lookup dictionaries for faster access
        dest_to_raw = {}  # Maps processed image path to raw source
        dest_to_info = {}  # Maps processed image path to pokemon_id and split
        
        # Build lookup tables and collect stats
        for raw_path, info in split_mapping.items():
            split = info['split']
            pokemon_id = info['pokemon_id']
            dest_path = info['dest_image']
            
            # Update stats
            split_stats[split]['images'] += 1
            split_stats[split]['pokemon'].add(pokemon_id + 1)  # Convert to 1-based ID
            
            # Build lookups
            dest_to_raw[dest_path] = raw_path
            dest_to_info[dest_path] = {'pokemon_id': pokemon_id, 'split': split}
            
        # Convert sets to counts for JSON serialization
        for split in split_stats:
            split_stats[split]['pokemon'] = len(split_stats[split]['pokemon'])
            
        logging.info("✅ Built lookup tables and collected stats")
        
        # Create dataset for each split
        datasets = {}
        for split in ['train', 'validation', 'test']:
            images_dir = dataset_path / split / "images"
            labels_dir = dataset_path / split / "labels"
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            image_files = sorted(images_dir.glob("*.jpg"))
            label_files = sorted(labels_dir.glob("*.txt"))
            
            if len(image_files) != len(label_files):
                raise ValueError(
                    f"Mismatched files in {split}: "
                    f"{len(image_files)} images, {len(label_files)} labels"
                )
            
            # Create dataset
            data = {
                "image": [],
                "label_file": [],
                "label": [],
                "raw_image": [],  # Track source image
                "pokemon_id": []  # Store 1-based Pokemon ID for reference
            }
            
            # Process images in parallel batches
            total_images = len(image_files)
            batch_size = 100  # Adjust based on available memory
            num_workers = min(mp.cpu_count(), 8)  # Use up to 8 workers
            
            logging.info(f"Processing {split} split ({total_images} images) using {num_workers} workers...")
            
            # Create batches
            batches = []
            for i in range(0, total_images, batch_size):
                batch_imgs = image_files[i:i + batch_size]
                batch_lbls = label_files[i:i + batch_size]
                batches.append((batch_imgs, batch_lbls, dataset_path, dest_to_raw, dest_to_info))
            
            # Process batches in parallel
            processed_total = 0
            with mp.Pool(num_workers) as pool:
                for batch_data, processed_count in pool.imap_unordered(self.process_image_batch, batches):
                    # Merge batch results
                    data["image"].extend(batch_data["image"])
                    data["label_file"].extend(batch_data["label_file"])
                    data["label"].extend(batch_data["label"])
                    data["raw_image"].extend(batch_data["raw_image"])
                    data["pokemon_id"].extend(batch_data["pokemon_id"])
                    
                    processed_total += processed_count
                    logging.info(f"  • Processed {processed_total}/{total_images} images ({processed_total/total_images*100:.1f}%)")
            
            # Create dataset with features and optimize memory usage
            logging.info(f"Creating {split} dataset...")
            
            # Create dataset directly with memory-mapped features
            datasets[split] = Dataset.from_dict(
                data,
                features=Features({
                    "image": Image(decode=False),  # Don't decode images in memory
                    "label": ClassLabel(num_classes=1025),
                    "label_file": Value("string"),
                    "raw_image": Value("string"),
                    "pokemon_id": Value("int32")
                })
            )
            
            # Enable memory mapping for large datasets
            datasets[split].set_format(
                type="numpy",
                columns=["image", "label"],
                output_all_columns=True
            )
            logging.info(f"✅ Created {split} dataset with {len(data['image'])} examples")
        
        # Create dataset dictionary
        dataset_dict = DatasetDict(datasets)
        
        # Create dataset card
        dataset_card = f"""
# Pokemon YOLO Dataset

## Dataset Summary
- **Total Pokemon**: 1025 (all generations)
- **Format**: YOLO detection format with full-image bounding boxes
- **Image Size**: 416x416 pixels
- **Class IDs**: 0-based indexing (0-1024)
- **Total Images**: {sum(split_stats[s]['images'] for s in ['train', 'validation', 'test'])}

## Splits
- **Train**: {split_stats['train']['images']} images ({split_stats['train']['pokemon']} Pokemon)
- **Validation**: {split_stats['validation']['images']} images ({split_stats['validation']['pokemon']} Pokemon)
- **Test**: {split_stats['test']['images']} images ({split_stats['test']['pokemon']} Pokemon)

## Format
- **Images**: 416x416 JPEG
- **Labels**: YOLO format (class_id x_center y_center width height)
- **Bounding Boxes**: Full-image (0.5 0.5 1.0 1.0)
- **Class IDs**: 0-based (0-1024)
- **Filename Format**: NNNN_XXX.jpg/txt where:
  - NNNN: 1-based Pokemon ID (0001-1025)
  - XXX: Sequential number for each Pokemon

## Split Strategy
- Per-Pokemon 70/15/15 split for Pokemon with enough images
- Pokemon with 6+ images: Standard 70/15/15 split
- Pokemon with 3-5 images: Minimal split (1 image per split)
- Pokemon with 1-2 images: All in training set
- No data leakage between splits

## Features
- **image**: PIL Image (416x416 JPEG)
- **label**: Class ID (0-1024)
- **label_file**: Path to YOLO format label file
- **raw_image**: Path to original source image
- **pokemon_id**: 1-based Pokemon ID (1-1025)

## Usage
```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("{dataset_name}")

# Access splits
train_ds = dataset["train"]
val_ds = dataset["validation"]
test_ds = dataset["test"]

# Access data
image = train_ds[0]["image"]  # PIL Image
label = train_ds[0]["label"]  # class ID (0-1024)
```

## License
CC BY-NC-SA 4.0

## Citation
Please cite both the original dataset and this version:

```bibtex
@misc{{pokemon-yolo-1025,
  author = {{Your Name}},
  title = {{Pokemon YOLO Dataset}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  howpublished = {{https://huggingface.co/datasets/{dataset_name}}}
}}
```
"""
        
        # Push to hub with optimized settings
        logging.info(f"Pushing dataset to {dataset_name}...")
        
        # First create the repo
        self.api.create_repo(
            repo_id=dataset_name,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
        
        # Update README
        self.api.upload_file(
            repo_id=dataset_name,
            repo_type="dataset",
            path_in_repo="README.md",
            path_or_fileobj=dataset_card.encode(),
        )
        
        # Push dataset with optimized settings and retry mechanism
        logging.info("Starting dataset upload (this may take a while)...")
        
        @backoff.on_exception(
            backoff.expo,
            (ConnectionError, NameResolutionError, RequestException),
            max_tries=10,  # Maximum number of retries
            max_time=7200,  # Maximum total time to retry (2 hours)
            giveup=lambda e: not retry_on_network_error(e),
            on_backoff=lambda details: logging.info(
                f"Retry {details['tries']}/10 after {details['wait']:.1f} seconds..."
            )
        )
        def push_with_retry():
            # Configure git for large file uploads
            git_configs = {
                "http.postBuffer": "512M",
                "http.lowSpeedLimit": "1000",
                "http.lowSpeedTime": "600",
                "http.maxRequestBuffer": "100M",
                "core.compression": "0",
                "http.timeout": "600"
            }
            for key, value in git_configs.items():
                subprocess.run(["git", "config", "--global", key, value], check=True)
            
            # Push dataset with optimized settings for large uploads
            dataset_dict.push_to_hub(
                repo_id=dataset_name,
                token=self.token,
                private=False,
                max_shard_size="200MB",  # Smaller shards for more reliable uploads
                embed_external_files=True,  # Embed images
                num_proc=2,  # Reduce concurrency to prevent timeouts
                commit_message="Upload Pokemon YOLO dataset with full-image bounding boxes",
                create_pr=False  # Direct push to main
            )
        
        # Execute with retry mechanism
        try:
            push_with_retry()
        except Exception as e:
            logging.error(f"Failed to upload dataset after all retries: {str(e)}")
            logging.info("You can resume the upload by running the script again.")
            raise
        
        logging.info(f"✅ Dataset uploaded to: https://huggingface.co/datasets/{dataset_name}")
        for split, stats in split_stats.items():
            logging.info(f"  • {split}: {stats['images']} images")

def main():
    """Main upload function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload Pokemon dataset to Hugging Face")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset")
    parser.add_argument("--dataset_name", required=True, help="Hugging Face dataset name")
    parser.add_argument("--token", help="Hugging Face token")
    
    args = parser.parse_args()
    
    uploader = PokemonDatasetUploader(args.token)
    uploader.create_yolo_dataset(args.dataset_path, args.dataset_name)

if __name__ == "__main__":
    main()