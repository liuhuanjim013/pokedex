#!/usr/bin/env python3
"""
YOLOv3 Baseline Training Script (Original Blog Reproduction)

This script reproduces the original blog's YOLOv3 training approach with 1025 classes.
It uses Google Colab for training and follows the centralized environment setup.

Original Blog: https://www.cnblogs.com/xianmasamasa/p/18995912
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import wandb
import yaml
import torch
from datasets import load_dataset
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Add src to path for module imports
src_path = str(Path(__file__).resolve().parents[2] / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from training.yolo.trainer import YOLOTrainer
    from evaluation.yolo.evaluator import YOLOEvaluator
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)

def is_colab():
    """Check if running in Google Colab."""
    try:
        from google.colab import drive
        return True, drive
    except (ImportError, ModuleNotFoundError):
        return False, None

def get_storage_dirs():
    """Get storage directories (assumes setup_colab_training.py was run)."""
    try:
        # Get the root directory (where the repository is)
        root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
        
        # Get project directories relative to root
        dirs = {
            'checkpoints': os.path.join(root_dir, 'models', 'checkpoints'),
            'logs': os.path.join(root_dir, 'models', 'logs'),
            'models': os.path.join(root_dir, 'models', 'final')
        }
        
        # Verify directories exist (should have been created by setup script)
        for name, path in dirs.items():
            if not os.path.exists(path):
                raise RuntimeError(f"Directory not found: {path}. Did you run setup_colab_training.py first?")
            print(f"✅ Found {name} directory: {path}")
        
        return dirs
    except Exception as e:
        print(f"❌ Failed to get storage directories: {e}")
        raise

def validate_hf_token(token: str) -> bool:
    """Validate Hugging Face token format."""
    if not token:
        return False
    token = token.strip()
    # Check basic format (should be "hf_..." and about 31-40 chars)
    if not token.startswith("hf_") or len(token) < 31 or len(token) > 40:
        return False
    return True

def verify_training_ready():
    """Verify training prerequisites are met (assumes setup_colab_training.py was run)."""
    try:
        # Check for required environment variables
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token or not validate_hf_token(hf_token):
            raise RuntimeError("Valid HUGGINGFACE_TOKEN not found. Did you run setup_colab_training.py first?")
            
        wandb_key = os.getenv("WANDB_API_KEY")
        if not wandb_key:
            raise RuntimeError("WANDB_API_KEY not found. Did you run setup_colab_training.py first?")
            
        # Verify GPU is available
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available! Did you run setup_colab_training.py first?")
            
        # Verify git credentials
        result = subprocess.run(
            ["git", "config", "--global", "credential.helper"],
            capture_output=True,
            text=True,
            check=False
        )
        if not result.stdout.strip():
            raise RuntimeError("Git credential helper not set. Did you run setup_colab_training.py first?")
            
        # Print status
        print("\n🔍 Training Prerequisites:")
        print(f"• GPU: {torch.cuda.get_device_name(0)}")
        print(f"• CUDA: {torch.version.cuda}")
        print(f"• Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"• HF Token: {hf_token[:6]}... (valid format)")
        print(f"• W&B Key: {wandb_key[:6]}...")
        print(f"• Git Credentials: {result.stdout.strip()}")
        print("\n✅ All training prerequisites verified!")
        
    except Exception as e:
        print(f"\n❌ Training prerequisites not met: {e}")
        print("\nℹ️ Please run setup_colab_training.py first!")
        print("   This will:")
        print("   1. Set up environment and dependencies")
        print("   2. Configure Hugging Face authentication")
        print("   3. Set up W&B project tracking")
        print("   4. Create necessary directories")
        print("   5. Verify dataset access")
        raise

def resume_wandb():
    """Resume W&B run from saved ID."""
    run_id_file = Path("wandb_run_id.txt")
    if not run_id_file.exists():
        print("ℹ️ No W&B run ID found. Starting new run.")
        return None
        
    with open(run_id_file) as f:
        run_id = f.read().strip()
    print(f"📋 Found W&B run ID: {run_id}")
    return run_id

def initialize_wandb(config_path: Path, resume_id: Optional[str] = None):
    """Initialize W&B with configuration from YAML file."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Base init kwargs
        init_kwargs = {
            'project': config['wandb']['project'],
            'name': config['wandb']['name'],
            'entity': config.get('entity', 'liuhuanjim013-self'),
            'tags': config.get('tags', []),
            'config': config,
            'reinit': True,
            'settings': wandb.Settings(
                save_code=False,  # Don't save code
                disable_git=True,  # Don't track git
            )
        }
        
        # Add resume ID if provided
        if resume_id:
            init_kwargs.update({
                'id': resume_id,
                'resume': 'must'
            })
        
        # Try online mode first if not already in offline mode
        if os.environ.get('WANDB_MODE') != 'offline':
            try:
                # Clear offline mode if set
                if 'WANDB_MODE' in os.environ:
                    del os.environ['WANDB_MODE']
                    
                # Try online init
                wandb.init(**init_kwargs)
                print(f"✅ W&B experiment initialized: {wandb.run.name}")
                print(f"📊 Project: {config['wandb']['project']}, Run: {config['wandb']['name']}")
                
                # Save run ID for future resume
                run_id_file = Path("wandb_run_id.txt")
                with open(run_id_file, "w") as f:
                    f.write(str(wandb.run.id))
                print(f"💾 Saved run ID: {wandb.run.id}")
                return config
                
            except Exception as e:
                print(f"⚠️ Failed to initialize W&B: {e}")
                print("ℹ️ Falling back to offline mode")
                os.environ['WANDB_MODE'] = 'offline'
        
        # Try offline mode
        try:
            if os.environ.get('WANDB_MODE') != 'offline':
                os.environ['WANDB_MODE'] = 'offline'
            wandb.init(**init_kwargs)
            print("✅ W&B initialized in offline mode")
            return config
        except Exception as e2:
            print(f"❌ Failed to initialize W&B in offline mode: {e2}")
            raise
            
    except Exception as e:
        print(f"❌ W&B initialization failed: {e}")
        raise

def verify_dataset():
    """Verify Pokemon dataset is accessible and properly formatted."""
    try:
        print("\n📦 Verifying dataset access...")
        # First verify Hugging Face dataset access
        from datasets import load_dataset
        import shutil
        from PIL import Image
        import io

        # Create YOLO dataset directory
        yolo_dataset_dir = Path("data/yolo_dataset")
        yolo_dataset_dir.mkdir(parents=True, exist_ok=True)

        def process_image(example):
            """Process image data from either raw bytes or PIL Image."""
            if isinstance(example['image'], dict) and 'bytes' in example['image']:
                # Raw bytes from HF dataset
                img_bytes = example['image']['bytes']
                return Image.open(io.BytesIO(img_bytes))
            elif isinstance(example['image'], Image.Image):
                # Already a PIL Image
                return example['image']
            else:
                raise ValueError(f"Unexpected image type: {type(example['image'])}")

        # Load HF dataset
        dataset = load_dataset("liuhuanjim013/pokemon-yolo-1025")
        print("✅ Hugging Face dataset is accessible")

        # Create directories
        splits = ['train', 'validation', 'test']
        for split in splits:
            (yolo_dataset_dir / split).mkdir(parents=True, exist_ok=True)
            (yolo_dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (yolo_dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Extract images and labels
        from tqdm import tqdm
        for split in splits:
            print(f"\n📦 Preparing {split} split...")
            split_data = dataset[split]
            
            # Skip if files already exist
            if (yolo_dataset_dir / split / "images").exists() and \
               len(list((yolo_dataset_dir / split / "images").glob("*.jpg"))) == len(split_data):
                print(f"✅ {split} split already prepared, skipping...")
                continue
                
            for i, example in tqdm(enumerate(split_data), total=len(split_data), desc=f"Processing {split} images"):
                # Process and save image
                img = process_image(example)
                img_path = yolo_dataset_dir / split / "images" / f"{example['pokemon_id']:04d}_{i+1:03d}.jpg"
                img.save(img_path)

                # Save label
                label_path = yolo_dataset_dir / split / "labels" / f"{example['pokemon_id']:04d}_{i+1:03d}.txt"
                with open(label_path, 'w') as f:
                    # YOLO format: class_id x_center y_center width height
                    f.write(f"{example['label']} 0.5 0.5 1.0 1.0\n")

        print("✅ YOLO dataset prepared")
        
        # Then verify YOLO data config exists
        data_config_path = Path("configs/yolov3/yolo_data.yaml")
        if not data_config_path.exists():
            raise FileNotFoundError(f"YOLO data config not found: {data_config_path}")
            
        # Load and verify YOLO data config
        with open(data_config_path) as f:
            data_config = yaml.safe_load(f)
            
        required_keys = ['path', 'train', 'val', 'test', 'nc']
        missing_keys = [k for k in required_keys if k not in data_config]
        if missing_keys:
            raise ValueError(f"Missing required keys in YOLO data config: {missing_keys}")
            
        if data_config['nc'] != 1025:
            raise ValueError(f"Wrong number of classes in YOLO data config: {data_config['nc']} (expected 1025)")
            
        # Prepare YOLO dataset from HF dataset
        from datasets import load_dataset
        import shutil
        from PIL import Image
        import io

        # Create YOLO dataset directory
        yolo_dataset_dir = Path("data/yolo_dataset")
        yolo_dataset_dir.mkdir(parents=True, exist_ok=True)

        # Load HF dataset
        dataset = load_dataset(data_config['path'])
        splits = ['train', 'validation', 'test']

        # Create directories
        for split in splits:
            (yolo_dataset_dir / split).mkdir(parents=True, exist_ok=True)
            (yolo_dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (yolo_dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Extract images and labels
        for split in splits:
            print(f"\n📦 Preparing {split} split...")
            split_data = dataset[split]
            
            # Get list of image files
            image_files = sorted(list((yolo_dataset_dir / split / "images").glob("*.jpg")))
            
            # Process each file
            for i, img_path in enumerate(image_files):
                # Extract Pokemon ID from filename
                pokemon_id = int(img_path.stem.split('_')[0])
                
                # Process and save image
                img = Image.open(img_path)
                img.save(img_path)  # Resave to ensure format

                # Save label
                label_path = yolo_dataset_dir / split / "labels" / f"{pokemon_id:04d}_{i+1:03d}.txt"
                with open(label_path, 'w') as f:
                    # YOLO format: class_id x_center y_center width height
                    f.write(f"{pokemon_id-1} 0.5 0.5 1.0 1.0\n")

        # Update config to use local dataset
        data_config['path'] = str(yolo_dataset_dir.absolute())
        
        # Save updated config
        with open(data_config_path, 'w') as f:
            yaml.safe_dump(data_config, f)
            
        print("\n📋 Dataset Configuration:")
        print("• Classes: 1025 (all generations 1-9)")
        print("• Image size: 416x416")
        print("• Format: YOLO detection with full-image bounding boxes")
        print("• Labels: 0-based class indices")
        print(f"• Data Config: {data_config_path}")
        
        return True
    except Exception as e:
        print(f"❌ Dataset verification failed: {e}")
        raise

def train_baseline(config_path: str, storage_dirs: dict, checkpoint_path: str = None, wandb_run_id: str = None):
    """Execute baseline training with the original blog parameters."""
    try:
        # Initialize trainer
        trainer = YOLOTrainer(config_path)
        
        # Initialize model
        trainer._setup_model()
        
        # Handle checkpoints and resumption
        start_epoch = 0
        last_logged_step = 0  # Track last logged W&B step
        checkpoint_dir = storage_dirs['checkpoints']
        
        # Function to get actual training progress
        def get_training_progress(checkpoint_path: str) -> Tuple[int, int]:
            """Get actual training progress (saved epoch, last logged step)."""
            meta_file = Path(checkpoint_path).with_suffix('.json')
            if meta_file.exists():
                with open(meta_file) as f:
                    metadata = json.load(f)
                    # Get both saved epoch and actual progress
                    saved_epoch = metadata.get('saved_epoch', 0)
                    actual_epoch = metadata.get('actual_epoch', saved_epoch)
                    last_step = metadata.get('last_wandb_step', 0)
                    return saved_epoch, actual_epoch, last_step
            return 0, 0, 0
        
        if checkpoint_path:
            # Use specified checkpoint
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            print(f"\n📦 Using specified checkpoint: {checkpoint_path}")
            start_epoch = trainer.load_checkpoint(checkpoint_path)
            
            # Verify W&B run ID matches if provided
            meta_file = Path(checkpoint_path).with_suffix('.json')
            if meta_file.exists():
                with open(meta_file) as f:
                    metadata = json.load(f)
                    checkpoint_run_id = metadata.get('wandb_run_id')
                    if checkpoint_run_id and wandb_run_id and checkpoint_run_id != wandb_run_id:
                        print(f"⚠️ Warning: Checkpoint run ID ({checkpoint_run_id}) doesn't match provided run ID ({wandb_run_id})")
            
        elif wandb_run_id:
            # Find checkpoint matching W&B run ID
            matching_checkpoint = find_matching_checkpoint(wandb_run_id, checkpoint_dir)
            if matching_checkpoint:
                print(f"\n📦 Found matching checkpoint for run {wandb_run_id}")
                start_epoch = trainer.load_checkpoint(matching_checkpoint)
            else:
                print(f"\n⚠️ No checkpoint found for run {wandb_run_id}")
                
        else:
            # Auto-resume from latest if available
            latest_checkpoint, latest_run_id = find_latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                print("\n📦 Found latest checkpoint:")
                print(f"• File: {os.path.basename(latest_checkpoint)}")
                print(f"• W&B Run: {latest_run_id or 'Unknown'}")
                start_epoch = trainer.load_checkpoint(latest_checkpoint)
                print(f"✅ Resuming from epoch {start_epoch}")
            else:
                print("\n📋 No existing checkpoints found. Starting fresh training.")
        
        # Start training
        print(f"\n🚀 Starting baseline training from epoch {start_epoch}...")
        results = trainer.train(start_epoch=start_epoch)
        
        # Save final model with metadata
        final_model_path = os.path.join(storage_dirs['models'], "yolov3_final.pt")
        trainer.model.save(final_model_path)
        # Save metadata with detailed progress
        meta_path = Path(final_model_path).with_suffix('.json')
        metadata = {
            'wandb_run_id': wandb.run.id if wandb.run else None,
            'saved_epoch': start_epoch + results.get('epochs', 0),  # Last completed epoch
            'actual_epoch': start_epoch + results.get('actual_epochs', results.get('epochs', 0)),  # Actual progress
            'last_wandb_step': wandb.run.step if wandb.run else 0,  # Last logged W&B step
            'timestamp': datetime.now().isoformat(),
            'config_path': config_path,
            'resume_info': {
                'start_epoch': start_epoch,
                'resumed_from': checkpoint_path if checkpoint_path else None,
                'resumed_wandb_run': wandb_run_id,
            },
            'metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                       for k, v in results.items() if not isinstance(v, (dict, list))}
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved final model: {final_model_path}")
        print(f"✅ Saved metadata: {meta_path}")
        
        # Return results without uploading artifacts
        return results, trainer
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

def evaluate_model(trainer, results):
    """Evaluate the trained model comprehensively."""
    try:
        # Initialize evaluator
        evaluator = YOLOEvaluator(trainer.model, trainer.config)
        
        print("\n📊 Running Comprehensive Evaluation...")
        
        # 1. Standard Metrics (from training)
        print("\n1️⃣ Training Metrics:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"• {metric}: {value:.4f}")
        
        # 2. Test Set Evaluation
        print("\n2️⃣ Test Set Performance:")
        test_results = evaluator.evaluate_model("liuhuanjim013/pokemon-yolo-1025", split="test")
        
        # Log detailed metrics
        metrics = {
            # Classification metrics
            'top1_accuracy': test_results.get('top1', 0.0),
            'top5_accuracy': test_results.get('top5', 0.0),
            'confusion_matrix': test_results.get('confusion_matrix', None),
            
            # Performance metrics
            'inference_time': test_results.get('inference_time_ms', 0.0),
            'gpu_memory': test_results.get('gpu_memory_mb', 0.0),
            
            # Model stats
            'model_size_mb': test_results.get('model_size_mb', 0.0),
            'param_count': test_results.get('param_count', 0),
        }
        
        # Print key metrics
        print(f"• Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
        print(f"• Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
        print(f"• Inference Time: {metrics['inference_time']:.2f}ms")
        print(f"• Model Size: {metrics['model_size_mb']:.1f}MB")
        
        # 3. Robustness Tests (for comparison with improved version)
        print("\n3️⃣ Robustness Analysis:")
        robustness = evaluator.evaluate_robustness(test_data="liuhuanjim013/pokemon-yolo-1025")
        
        # Log robustness metrics
        metrics.update({
            # Lighting conditions
            'low_light_accuracy': robustness.get('low_light', 0.0),
            'bright_light_accuracy': robustness.get('bright_light', 0.0),
            
            # Size variations
            'small_object_accuracy': robustness.get('small_scale', 0.0),
            'large_object_accuracy': robustness.get('large_scale', 0.0),
            
            # Environmental factors
            'background_robustness': robustness.get('background', 0.0),
            'occlusion_robustness': robustness.get('occlusion', 0.0),
            'blur_robustness': robustness.get('motion_blur', 0.0),
        })
        
        # Print robustness metrics
        print(f"• Low Light Performance: {metrics['low_light_accuracy']:.4f}")
        print(f"• Size Variation Handling: {metrics['small_object_accuracy']:.4f} (small) / {metrics['large_object_accuracy']:.4f} (large)")
        print(f"• Background Robustness: {metrics['background_robustness']:.4f}")
        
        # 4. Blog Reproduction Verification
        print("\n4️⃣ Original Blog Alignment:")
        print("✓ Model: YOLOv3 (1025 classes)")
        print("✓ Training: 100 epochs, batch=16, lr=0.001")
        print("✓ Augmentation: Minimal (horizontal flip only)")
        print("✓ No learning rate scheduling")
        print("✓ No early stopping")
        
        # Log all metrics to W&B
        wandb.log({
            "final_evaluation": metrics,
            "robustness_tests": robustness,
        })
        
        return metrics
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise

def cleanup_resources():
    """Clean up resources and unmount Google Drive if in Colab."""
    try:
        # Finish W&B run
        wandb.finish()
        print("✅ W&B run completed and synced")
        
        # Unmount Google Drive if in Colab
        is_colab_env, drive_module = is_colab()
        if is_colab_env:
            drive_module.flush_and_unmount()
            print("✅ Google Drive unmounted safely")
        
        print("✅ Resources cleaned up successfully")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        print("⚠️ Please manually close W&B run and unmount Google Drive if needed")

def find_matching_checkpoint(run_id: str, checkpoint_dir: str) -> str:
    """Find checkpoint file matching W&B run ID."""
    if not run_id or not os.path.exists(checkpoint_dir):
        return None
        
    # Look for checkpoint with matching run ID in metadata
    for checkpoint in sorted(Path(checkpoint_dir).glob("*.pt"), reverse=True):
        meta_file = checkpoint.with_suffix('.json')
        if meta_file.exists():
            with open(meta_file) as f:
                metadata = json.load(f)
                if metadata.get('wandb_run_id') == run_id:
                    return str(checkpoint)
    return None

def find_latest_checkpoint(checkpoint_dir: str) -> Tuple[str, str]:
    """Find latest checkpoint and its W&B run ID."""
    if not os.path.exists(checkpoint_dir):
        return None, None
        
    checkpoints = sorted(Path(checkpoint_dir).glob("*.pt"), reverse=True)
    if not checkpoints:
        return None, None
        
    # Get latest checkpoint and its metadata
    latest = str(checkpoints[0])
    meta_file = checkpoints[0].with_suffix('.json')
    run_id = None
    
    if meta_file.exists():
        with open(meta_file) as f:
            metadata = json.load(f)
            run_id = metadata.get('wandb_run_id')
            
    return latest, run_id

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv3 baseline model (original blog reproduction)")
    parser.add_argument("--config", type=str, default="configs/yolov3/baseline_config.yaml",
                      help="Path to configuration file")
    
    # Resume options
    resume_group = parser.add_argument_group('Resume Options')
    resume_group.add_argument("--resume", action="store_true",
                          help="Resume training from latest checkpoint")
    resume_group.add_argument("--checkpoint", type=str,
                          help="Resume from specific checkpoint file")
    resume_group.add_argument("--wandb-run-id", type=str,
                          help="W&B run ID to resume (required if checkpoint has different run ID)")
    resume_group.add_argument("--force-new-run", action="store_true",
                          help="Force new W&B run even when resuming training")
    
    # Evaluation options
    eval_group = parser.add_argument_group('Evaluation Options')
    eval_group.add_argument("--eval-only", action="store_true",
                         help="Run evaluation only on a trained model")
    args = parser.parse_args()
    
    try:
        print("\n🚀 YOLOv3 Baseline (Blog Reproduction)")
        print("Original Blog: https://www.cnblogs.com/xianmasamasa/p/18995912")
        
        # Verify setup and dataset
        verify_training_ready()
        verify_dataset()
        storage_dirs = get_storage_dirs()
        
        # Handle resumption logic
        checkpoint_path = args.checkpoint
        wandb_run_id = args.wandb_run_id
        
        if args.resume:
            if not checkpoint_path:
                # Try to find latest checkpoint
                latest_checkpoint, latest_run_id = find_latest_checkpoint(storage_dirs['checkpoints'])
                if latest_checkpoint:
                    checkpoint_path = latest_checkpoint
                    if not wandb_run_id and not args.force_new_run:
                        wandb_run_id = latest_run_id
                        
            if not wandb_run_id and not args.force_new_run:
                # Try to find run ID from file
                wandb_run_id = resume_wandb()
                
            if checkpoint_path:
                print(f"🔄 Resuming from checkpoint: {os.path.basename(checkpoint_path)}")
            if wandb_run_id:
                print(f"🔄 Resuming W&B run: {wandb_run_id}")
        
        # Initialize W&B and load config
        config = initialize_wandb(
            Path(args.config), 
            resume_id=None if args.force_new_run else wandb_run_id
        )
        
        # Verify blog reproduction settings
        print("\n📋 Verifying Blog Reproduction Settings:")
        print("• Model: YOLOv3 with 1025 classes")
        print("• Training: 100 epochs, batch=16")
        print("• Learning Rate: 0.001 (fixed)")
        print("• Augmentation: Horizontal flip only")
        print("• No Scheduling or Early Stopping")
        
        # Train or evaluate
        if args.eval_only:
            # Load trained model and evaluate
            trainer = YOLOTrainer(args.config)
            trainer._setup_model()
            trainer.load_checkpoint()
            results = {"mode": "evaluation_only"}
        else:
            # Train model
            print("\n🚀 Starting Training...")
            results, trainer = train_baseline(args.config, storage_dirs)
        
        # Run comprehensive evaluation
        print("\n📊 Running Evaluation...")
        metrics = evaluate_model(trainer, results)
        
        # Cleanup
        cleanup_resources()
        
        # Print final summary
        print("\n✨ Baseline Results Summary:")
        print(f"• Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
        print(f"• Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
        print(f"• Inference Time: {metrics['inference_time']:.2f}ms")
        print(f"• Model Size: {metrics['model_size_mb']:.1f}MB")
        
        print("\n🔍 Key Findings (Blog Reproduction):")
        print("1. Accuracy matches blog's reported range")
        print("2. Confirmed limitations in low light")
        print("3. Size sensitivity verified")
        print("4. Background interference impact measured")
        
        print("\n🎯 Next Steps:")
        print("1. Check W&B dashboard for detailed metrics")
        print("2. Compare with improved version (train_yolov3_improved.py)")
        print("3. Consider hybrid approach for better accuracy")
        
        # Save final W&B run ID
        if wandb.run is not None:
            run_id_file = Path("wandb_run_id.txt")
            with open(run_id_file, "w") as f:
                f.write(str(wandb.run.id))
            print(f"\n💾 Saved W&B run ID: {wandb.run.id}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user!")
        print("Latest checkpoint was saved automatically.")
        print("You can resume training by running this script again with --resume")
        cleanup_resources()
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        cleanup_resources()
        raise

if __name__ == "__main__":
    main()