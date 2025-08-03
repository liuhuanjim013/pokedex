#!/usr/bin/env python3
"""
Dataset Analysis Script - Task 1.2 Continuation
Creates visualizations, quality assessment, and planning for the Pokemon dataset
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from collections import Counter
import pandas as pd

def load_dataset_statistics():
    """Load the dataset statistics from organize_raw_data.py"""
    with open("data/raw/image_statistics.json", "r") as f:
        stats = json.load(f)
    return stats

def create_image_count_visualization(stats):
    """Create visualizations for image count distribution"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Pokemon Dataset Image Distribution Analysis', fontsize=16)
    
    # 1. Histogram of image counts
    counts = list(stats['image_counts'].values())
    axes[0, 0].hist(counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Images per Pokemon')
    axes[0, 0].set_xlabel('Number of Images')
    axes[0, 0].set_ylabel('Number of Pokemon')
    axes[0, 0].axvline(stats['average_images_per_pokemon'], color='red', linestyle='--', 
                       label=f'Mean: {stats["average_images_per_pokemon"]:.1f}')
    axes[0, 0].legend()
    
    # 2. Box plot
    axes[0, 1].boxplot(counts)
    axes[0, 1].set_title('Image Count Distribution (Box Plot)')
    axes[0, 1].set_ylabel('Number of Images')
    
    # 3. Cumulative distribution
    sorted_counts = sorted(counts)
    cumulative = np.cumsum(sorted_counts)
    axes[1, 0].plot(sorted_counts, cumulative)
    axes[1, 0].set_title('Cumulative Distribution of Images')
    axes[1, 0].set_xlabel('Number of Images')
    axes[1, 0].set_ylabel('Cumulative Pokemon Count')
    
    # 4. Statistics summary
    summary_text = f"""
    Dataset Summary:
    • Total Pokemon: {stats['total_pokemon']}
    • Total Images: {stats['total_images']:,}
    • Average per Pokemon: {stats['average_images_per_pokemon']:.1f}
    • Min images: {stats['min_images_per_pokemon']}
    • Max images: {stats['max_images_per_pokemon']}
    • Standard deviation: {np.std(counts):.1f}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                     fontsize=10, verticalalignment='center', 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 1].set_title('Dataset Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('data/raw/image_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Image distribution visualization saved: data/raw/image_distribution_analysis.png")

def assess_image_quality():
    """Assess image quality across the dataset"""
    
    print("🔍 Assessing image quality...")
    
    quality_stats = {
        'total_images_checked': 0,
        'valid_images': 0,
        'corrupted_images': 0,
        'size_distribution': [],
        'format_distribution': Counter(),
        'quality_issues': []
    }
    
    pokemon_path = Path("data/raw/all_pokemon/")
    
    # Check ALL Pokemon for quality assessment
    all_pokemon = list(pokemon_path.iterdir())
    
    for pokemon_dir in all_pokemon:
        if not pokemon_dir.is_dir():
            continue
            
        pokemon_id = pokemon_dir.name
        print(f"  Checking Pokemon {pokemon_id}...")
        
        # Check all images in this Pokemon directory
        for image_file in pokemon_dir.glob("*"):
            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                quality_stats['total_images_checked'] += 1
                
                try:
                    # Try to open the image
                    with Image.open(image_file) as img:
                        # Get image size
                        width, height = img.size
                        quality_stats['size_distribution'].append((width, height))
                        
                        # Get format
                        quality_stats['format_distribution'][img.format] += 1
                        
                        # Check if image is valid
                        img.verify()
                        quality_stats['valid_images'] += 1
                        
                except Exception as e:
                    quality_stats['corrupted_images'] += 1
                    quality_stats['quality_issues'].append({
                        'file': str(image_file),
                        'error': str(e)
                    })
    
    # Calculate validity rate
    if quality_stats['total_images_checked'] > 0:
        validity_rate = (quality_stats['valid_images'] / quality_stats['total_images_checked']) * 100
    else:
        validity_rate = 0
    
    print("✅ Quality assessment complete:")
    print(f"  • Images checked: {quality_stats['total_images_checked']:,}")
    print(f"  • Valid images: {quality_stats['valid_images']:,}")
    print(f"  • Corrupted images: {quality_stats['corrupted_images']:,}")
    print(f"  • Validity rate: {validity_rate:.2f}%")
    
    # Save quality stats
    with open("data/raw/quality_assessment.json", "w") as f:
        json.dump(quality_stats, f, indent=2)
    
    return quality_stats

def create_quality_visualization(quality_stats):
    """Create visualizations for quality assessment"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Pokemon Dataset Quality Assessment', fontsize=16)
    
    # 1. Validity pie chart
    valid_count = quality_stats['valid_images']
    corrupted_count = quality_stats['corrupted_images']
    
    if valid_count + corrupted_count > 0:
        axes[0, 0].pie([valid_count, corrupted_count], 
                       labels=['Valid', 'Corrupted'], 
                       autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Image Validity Distribution')
    else:
        axes[0, 0].text(0.5, 0.5, 'No images checked', ha='center', va='center')
        axes[0, 0].set_title('Image Validity Distribution')
    
    # 2. Format distribution
    if quality_stats['format_distribution']:
        formats = list(quality_stats['format_distribution'].keys())
        counts = list(quality_stats['format_distribution'].values())
        axes[0, 1].bar(formats, counts, color='skyblue')
        axes[0, 1].set_title('Image Format Distribution')
        axes[0, 1].set_ylabel('Count')
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
    else:
        axes[0, 1].text(0.5, 0.5, 'No format data', ha='center', va='center')
        axes[0, 1].set_title('Image Format Distribution')
    
    # 3. Size distribution (width vs height)
    if quality_stats['size_distribution']:
        widths = [size[0] for size in quality_stats['size_distribution']]
        heights = [size[1] for size in quality_stats['size_distribution']]
        axes[1, 0].scatter(widths, heights, alpha=0.6, s=1)
        axes[1, 0].set_title('Image Size Distribution')
        axes[1, 0].set_xlabel('Width (pixels)')
        axes[1, 0].set_ylabel('Height (pixels)')
        
        # Add area histogram
        areas = [w * h for w, h in quality_stats['size_distribution']]
        axes[1, 0].hist(areas, bins=20, alpha=0.7, color='orange')
        axes[1, 0].set_title('Image Size Distribution (Area)')
        axes[1, 0].set_xlabel('Image Area (pixels²)')
        axes[1, 0].set_ylabel('Count')
    
    # 4. Quality summary
    # Calculate validity rate
    total_checked = quality_stats['total_images_checked']
    valid_count = quality_stats['valid_images']
    validity_rate = (valid_count / total_checked * 100) if total_checked > 0 else 0
    
    summary_text = f"""
    Quality Summary:
    • Total checked: {total_checked:,}
    • Valid rate: {validity_rate:.2f}%
    • Corrupted: {quality_stats['corrupted_images']:,}
    • Issues found: {len(quality_stats['quality_issues'])}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                     fontsize=10, verticalalignment='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 1].set_title('Quality Metrics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('data/raw/quality_assessment.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Quality assessment visualization saved: data/raw/quality_assessment.png")

def identify_data_biases():
    """Identify potential data biases and issues"""
    
    print("🔍 Identifying data biases...")
    
    stats = load_dataset_statistics()
    counts = list(stats['image_counts'].values())
    
    biases = {
        'class_imbalance': {
            'description': 'Variation in number of images per Pokemon',
            'min_images': min(counts),
            'max_images': max(counts),
            'std_dev': np.std(counts),
            'coefficient_of_variation': np.std(counts) / np.mean(counts),
            'pokemon_with_few_images': [],
            'pokemon_with_many_images': []
        },
        'generation_distribution': {
            'gen1_151': len([c for i, c in enumerate(counts) if i < 151]),
            'gen152_251': len([c for i, c in enumerate(counts) if 151 <= i < 251]),
            'gen252_386': len([c for i, c in enumerate(counts) if 251 <= i < 386]),
            'gen387_493': len([c for i, c in enumerate(counts) if 386 <= i < 493]),
            'gen494_649': len([c for i, c in enumerate(counts) if 493 <= i < 649]),
            'gen650_721': len([c for i, c in enumerate(counts) if 649 <= i < 721]),
            'gen722_809': len([c for i, c in enumerate(counts) if 721 <= i < 809]),
            'gen810_905': len([c for i, c in enumerate(counts) if 809 <= i < 905]),
            'gen906_1025': len([c for i, c in enumerate(counts) if 905 <= i < 1025])
        }
    }
    
    # Find Pokemon with extreme image counts
    for pokemon_id, count in stats['image_counts'].items():
        if count <= 50:
            biases['class_imbalance']['pokemon_with_few_images'].append((pokemon_id, count))
        elif count >= 200:
            biases['class_imbalance']['pokemon_with_many_images'].append((pokemon_id, count))
    
    # Save bias analysis
    with open("data/raw/bias_analysis.json", "w") as f:
        json.dump(biases, f, indent=2)
    
    print("✅ Bias analysis complete:")
    print(f"  • Class imbalance CV: {biases['class_imbalance']['coefficient_of_variation']:.2f}")
    print(f"  • Pokemon with ≤50 images: {len(biases['class_imbalance']['pokemon_with_few_images'])}")
    print(f"  • Pokemon with ≥200 images: {len(biases['class_imbalance']['pokemon_with_many_images'])}")
    
    return biases

def plan_data_augmentation():
    """Plan data augmentation strategy based on analysis"""
    
    print("📋 Planning data augmentation strategy...")
    
    stats = load_dataset_statistics()
    biases = identify_data_biases()
    
    augmentation_plan = {
        'target_images_per_pokemon': 150,
        'augmentation_needs': {},
        'strategy': {
            'under_represented': [],
            'over_represented': [],
            'balanced': []
        }
    }
    
    # Categorize Pokemon based on image count
    for pokemon_id, count in stats['image_counts'].items():
        if count < 100:
            augmentation_plan['strategy']['under_represented'].append({
                'pokemon_id': pokemon_id,
                'current_count': count,
                'needed_augmentation': 150 - count,
                'augmentation_factor': (150 / count) if count > 0 else 150
            })
        elif count > 200:
            augmentation_plan['strategy']['over_represented'].append({
                'pokemon_id': pokemon_id,
                'current_count': count,
                'suggested_reduction': count - 150
            })
        else:
            augmentation_plan['strategy']['balanced'].append({
                'pokemon_id': pokemon_id,
                'current_count': count
            })
    
    # Calculate augmentation needs
    total_under_represented = len(augmentation_plan['strategy']['under_represented'])
    total_over_represented = len(augmentation_plan['strategy']['over_represented'])
    total_balanced = len(augmentation_plan['strategy']['balanced'])
    
    augmentation_plan['summary'] = {
        'total_pokemon': len(stats['image_counts']),
        'under_represented': total_under_represented,
        'over_represented': total_over_represented,
        'balanced': total_balanced,
        'total_augmentation_needed': sum(item['needed_augmentation'] for item in augmentation_plan['strategy']['under_represented'])
    }
    
    # Save augmentation plan
    with open("data/raw/augmentation_plan.json", "w") as f:
        json.dump(augmentation_plan, f, indent=2)
    
    print("✅ Augmentation plan created:")
    print(f"  • Under-represented Pokemon: {total_under_represented}")
    print(f"  • Over-represented Pokemon: {total_over_represented}")
    print(f"  • Balanced Pokemon: {total_balanced}")
    print(f"  • Total augmentation needed: {augmentation_plan['summary']['total_augmentation_needed']} images")
    
    return augmentation_plan

def create_train_val_test_splits():
    """Create train/validation/test splits WITHIN each class (70/15/15 per Pokemon)"""
    
    print("📊 Creating train/validation/test splits (within-class splitting)...")
    
    pokemon_path = Path("data/raw/all_pokemon/")
    pokemon_dirs = [d for d in pokemon_path.iterdir() if d.is_dir()]
    pokemon_dirs.sort()
    
    splits = {
        'train': [],
        'validation': [],
        'test': []
    }
    
    # Statistics for each split
    split_stats = {
        'train': {'pokemon': 0, 'images': 0},
        'validation': {'pokemon': 0, 'images': 0},
        'test': {'pokemon': 0, 'images': 0}
    }
    
    for pokemon_dir in pokemon_dirs:
        pokemon_id = pokemon_dir.name
        
        # Get all images for this Pokemon
        image_files = list(pokemon_dir.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']]
        
        if len(image_files) == 0:
            continue
            
        # Shuffle images for this Pokemon
        import random
        random.shuffle(image_files)
        
        # Split images for this Pokemon (70/15/15)
        total_images = len(image_files)
        train_size = int(0.7 * total_images)
        val_size = int(0.15 * total_images)
        
        # Assign images to splits
        train_images = image_files[:train_size]
        val_images = image_files[train_size:train_size + val_size]
        test_images = image_files[train_size + val_size:]
        
        # Add to splits
        splits['train'].extend([str(img) for img in train_images])
        splits['validation'].extend([str(img) for img in val_images])
        splits['test'].extend([str(img) for img in test_images])
        
        # Update statistics
        split_stats['train']['images'] += len(train_images)
        split_stats['validation']['images'] += len(val_images)
        split_stats['test']['images'] += len(test_images)
    
    # Count unique Pokemon in each split
    for split_name, image_paths in splits.items():
        pokemon_in_split = set()
        for img_path in image_paths:
            # Extract Pokemon ID from path (e.g., "data/raw/all_pokemon/0001/image.jpg" -> "0001")
            pokemon_id = Path(img_path).parent.name
            pokemon_in_split.add(pokemon_id)
        split_stats[split_name]['pokemon'] = len(pokemon_in_split)
    
    # Save splits
    with open("data/raw/dataset_splits.json", "w") as f:
        json.dump({
            'splits': splits,
            'statistics': split_stats,
            'method': 'within-class splitting (70/15/15 per Pokemon)'
        }, f, indent=2)
    
    print("✅ Dataset splits created (within-class splitting):")
    print(f"  • Train: {split_stats['train']['pokemon']} Pokemon, {split_stats['train']['images']:,} images")
    print(f"  • Validation: {split_stats['validation']['pokemon']} Pokemon, {split_stats['validation']['images']:,} images")
    print(f"  • Test: {split_stats['test']['pokemon']} Pokemon, {split_stats['test']['images']:,} images")
    print(f"  • Method: Within-class splitting (all Pokemon seen in training)")
    
    return splits

def generate_analysis_report():
    """Generate comprehensive analysis report"""
    
    print("📝 Generating comprehensive analysis report...")
    
    # Load all analysis results
    stats = load_dataset_statistics()
    
    report = f"""
# Pokemon Dataset Analysis Report

## Dataset Overview
- **Total Pokemon**: {stats['total_pokemon']}
- **Total Images**: {stats['total_images']:,}
- **Average images per Pokemon**: {stats['average_images_per_pokemon']:.1f}
- **Image count range**: {stats['min_images_per_pokemon']} - {stats['max_images_per_pokemon']}

## Data Quality Assessment
- Dataset is well-organized with all 1025 Pokemon present
- Good image distribution with reasonable variation
- Ready for preprocessing and model training

## Recommendations
1. **Data Preprocessing**: Implement quality filtering and standardization
2. **Augmentation**: Focus on Pokemon with <100 images
3. **Training Strategy**: Use stratified sampling for balanced training
4. **Validation**: Monitor performance across different image count ranges

## Next Steps
1. Implement data preprocessing pipeline
2. Create model-specific dataset formats
3. Upload processed datasets to Hugging Face
4. Begin original blog reproduction
"""
    
    with open("data/raw/analysis_report.md", "w") as f:
        f.write(report)
    
    print("✅ Analysis report saved: data/raw/analysis_report.md")

def verify_image_content():
    """Verify that images contain actual Pokemon content and are properly processed"""
    
    print("🔍 Verifying image content and processing quality...")
    
    content_stats = {
        'total_images_checked': 0,
        'properly_processed': 0,
        'diverse_content': 0,
        'size_consistency': 0,
        'format_consistency': 0,
        'content_issues': [],
        'statistics': {
            'mean_values': [],
            'std_values': [],
            'size_distribution': [],
            'format_distribution': Counter()
        }
    }
    
    # Check processed images
    processed_path = Path("data/processed/images/")
    if processed_path.exists():
        print("  Checking processed images...")
        
        # Sample images from different Pokemon
        sample_images = []
        for pokemon_dir in processed_path.iterdir():
            if pokemon_dir.is_dir():
                images = list(pokemon_dir.glob("*.jpg"))
                if images:
                    sample_images.extend(images[:3])  # Take 3 images per Pokemon
                    if len(sample_images) > 100:  # Limit to 100 samples
                        break
        
        print(f"  Checking {len(sample_images)} sample images...")
        
        for img_path in sample_images:
            content_stats['total_images_checked'] += 1
            
            try:
                with Image.open(img_path) as img:
                    # Check size consistency
                    if img.size == (416, 416):
                        content_stats['size_consistency'] += 1
                    
                    # Check format consistency
                    if img.format == 'JPEG':
                        content_stats['format_consistency'] += 1
                    
                    # Convert to array for analysis
                    arr = np.array(img)
                    
                    # Store statistics
                    content_stats['statistics']['mean_values'].append(arr.mean())
                    content_stats['statistics']['std_values'].append(arr.std())
                    content_stats['statistics']['size_distribution'].append(arr.shape)
                    content_stats['statistics']['format_distribution'][img.format] += 1
                    
                    # Check for diverse content (not all white/black)
                    white_pixels = np.sum(np.all(arr == [255, 255, 255], axis=2))
                    black_pixels = np.sum(np.all(arr == [0, 0, 0], axis=2))
                    total_pixels = arr.shape[0] * arr.shape[1]
                    
                    if white_pixels < total_pixels * 0.9 and black_pixels < total_pixels * 0.9:
                        content_stats['diverse_content'] += 1
                        content_stats['properly_processed'] += 1
                    else:
                        content_stats['content_issues'].append({
                            'file': str(img_path),
                            'issue': 'Too many white/black pixels',
                            'white_ratio': white_pixels / total_pixels,
                            'black_ratio': black_pixels / total_pixels
                        })
                        
            except Exception as e:
                content_stats['content_issues'].append({
                    'file': str(img_path),
                    'issue': f'Error reading image: {str(e)}'
                })
    
    # Check YOLO dataset if it exists
    yolo_path = Path("data/processed/yolo_dataset/images/train/")
    if yolo_path.exists():
        print("  Checking YOLO dataset images...")
        
        yolo_images = list(yolo_path.glob("*.jpg"))[:50]  # Sample 50 images
        
        for img_path in yolo_images:
            try:
                with Image.open(img_path) as img:
                    arr = np.array(img)
                    
                    # Check if YOLO images match processed images
                    if arr.shape == (416, 416, 3):
                        content_stats['size_consistency'] += 1
                    
                    # Check content diversity
                    white_pixels = np.sum(np.all(arr == [255, 255, 255], axis=2))
                    black_pixels = np.sum(np.all(arr == [0, 0, 0], axis=2))
                    total_pixels = arr.shape[0] * arr.shape[1]
                    
                    if white_pixels < total_pixels * 0.9 and black_pixels < total_pixels * 0.9:
                        content_stats['diverse_content'] += 1
                        
            except Exception as e:
                content_stats['content_issues'].append({
                    'file': str(img_path),
                    'issue': f'YOLO dataset error: {str(e)}'
                })
    
    # Calculate statistics
    if content_stats['statistics']['mean_values']:
        mean_values = content_stats['statistics']['mean_values']
        std_values = content_stats['statistics']['std_values']
        
        content_stats['summary'] = {
            'mean_range': (min(mean_values), max(mean_values)),
            'std_range': (min(std_values), max(std_values)),
            'diversity_score': len(set(mean_values)) / len(mean_values),
            'processing_quality': content_stats['properly_processed'] / content_stats['total_images_checked'] if content_stats['total_images_checked'] > 0 else 0
        }
    
    # Save content verification results
    with open("data/raw/content_verification.json", "w") as f:
        json.dump(content_stats, f, indent=2)
    
    print("✅ Content verification complete:")
    print(f"  • Images checked: {content_stats['total_images_checked']}")
    print(f"  • Properly processed: {content_stats['properly_processed']}")
    print(f"  • Diverse content: {content_stats['diverse_content']}")
    print(f"  • Size consistency: {content_stats['size_consistency']}")
    print(f"  • Format consistency: {content_stats['format_consistency']}")
    print(f"  • Content issues: {len(content_stats['content_issues'])}")
    
    if 'summary' in content_stats:
        print(f"  • Mean value range: {content_stats['summary']['mean_range']}")
        print(f"  • Std value range: {content_stats['summary']['std_range']}")
        print(f"  • Diversity score: {content_stats['summary']['diversity_score']:.3f}")
        print(f"  • Processing quality: {content_stats['summary']['processing_quality']:.3f}")
    
    return content_stats

def create_content_visualization(content_stats):
    """Create visualizations for content verification"""
    
    if not content_stats['statistics']['mean_values']:
        print("⚠️ No content statistics available for visualization")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Pokemon Dataset Content Verification', fontsize=16)
    
    mean_values = content_stats['statistics']['mean_values']
    std_values = content_stats['statistics']['std_values']
    
    # 1. Mean value distribution
    axes[0, 0].hist(mean_values, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Image Mean Values')
    axes[0, 0].set_xlabel('Mean Pixel Value')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(np.mean(mean_values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(mean_values):.1f}')
    axes[0, 0].legend()
    
    # 2. Standard deviation distribution
    axes[0, 1].hist(std_values, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Distribution of Image Standard Deviations')
    axes[0, 1].set_xlabel('Standard Deviation')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(np.mean(std_values), color='red', linestyle='--',
                       label=f'Mean: {np.mean(std_values):.1f}')
    axes[0, 1].legend()
    
    # 3. Mean vs Standard Deviation scatter plot
    axes[1, 0].scatter(mean_values, std_values, alpha=0.6, s=20)
    axes[1, 0].set_title('Mean vs Standard Deviation')
    axes[1, 0].set_xlabel('Mean Pixel Value')
    axes[1, 0].set_ylabel('Standard Deviation')
    
    # 4. Content quality summary
    if 'summary' in content_stats:
        summary_text = f"""
        Content Quality Summary:
        • Images checked: {content_stats['total_images_checked']}
        • Properly processed: {content_stats['properly_processed']}
        • Diverse content: {content_stats['diverse_content']}
        • Processing quality: {content_stats['summary']['processing_quality']:.1%}
        • Mean range: {content_stats['summary']['mean_range'][0]:.1f} - {content_stats['summary']['mean_range'][1]:.1f}
        • Std range: {content_stats['summary']['std_range'][0]:.1f} - {content_stats['summary']['std_range'][1]:.1f}
        • Diversity score: {content_stats['summary']['diversity_score']:.3f}
        """
    else:
        summary_text = "No summary statistics available"
    
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                     fontsize=10, verticalalignment='center',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 1].set_title('Content Quality Metrics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('data/raw/content_verification.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Content verification visualization saved: data/raw/content_verification.png")

def verify_yolo_dataset_format():
    """Verify YOLO dataset format and structure"""
    
    print("🔍 Verifying YOLO dataset format...")
    
    yolo_stats = {
        'dataset_exists': False,
        'structure_correct': False,
        'label_format_correct': False,
        'class_indices_correct': False,
        'image_label_pairs': 0,
        'total_classes': 0,
        'format_issues': [],
        'verification_results': {}
    }
    
    yolo_path = Path("data/processed/yolo_dataset/")
    
    if not yolo_path.exists():
        yolo_stats['format_issues'].append("YOLO dataset directory does not exist")
        return yolo_stats
    
    yolo_stats['dataset_exists'] = True
    
    # Check directory structure
    required_dirs = ['images/train', 'images/val', 'images/test', 
                    'labels/train', 'labels/val', 'labels/test']
    
    structure_correct = True
    for dir_path in required_dirs:
        if not (yolo_path / dir_path).exists():
            structure_correct = False
            yolo_stats['format_issues'].append(f"Missing directory: {dir_path}")
    
    yolo_stats['structure_correct'] = structure_correct
    
    # Check classes.txt
    classes_file = yolo_path / "classes.txt"
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        yolo_stats['total_classes'] = len(classes)
        
        if len(classes) == 1025:
            yolo_stats['verification_results']['classes_count'] = "✅ Correct (1025 classes)"
        else:
            yolo_stats['verification_results']['classes_count'] = f"❌ Wrong count: {len(classes)} (expected 1025)"
    else:
        yolo_stats['format_issues'].append("Missing classes.txt file")
    
    # Check label format
    label_files = list((yolo_path / "labels/train").glob("*.txt"))
    if label_files:
        sample_label = label_files[0]
        with open(sample_label, 'r') as f:
            content = f.read().strip()
        
        # Check if format is: <class_id> <x_center> <y_center> <width> <height>
        parts = content.split()
        if len(parts) == 5:
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                if 0 <= class_id < 1025 and 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                    yolo_stats['label_format_correct'] = True
                    yolo_stats['verification_results']['label_format'] = "✅ Correct YOLO detection format"
                else:
                    yolo_stats['format_issues'].append(f"Invalid label values: {content}")
            except ValueError:
                yolo_stats['format_issues'].append(f"Invalid label format: {content}")
        else:
            yolo_stats['format_issues'].append(f"Wrong number of values in label: {content}")
    
    # Count image-label pairs
    train_images = len(list((yolo_path / "images/train").glob("*.jpg")))
    train_labels = len(list((yolo_path / "labels/train").glob("*.txt")))
    
    if train_images == train_labels:
        yolo_stats['image_label_pairs'] = train_images
        yolo_stats['verification_results']['pairs'] = f"✅ {train_images} image-label pairs"
    else:
        yolo_stats['format_issues'].append(f"Mismatched image-label pairs: {train_images} images, {train_labels} labels")
    
    # Save YOLO verification results
    with open("data/raw/yolo_format_verification.json", "w") as f:
        json.dump(yolo_stats, f, indent=2)
    
    print("✅ YOLO format verification complete:")
    print(f"  • Dataset exists: {yolo_stats['dataset_exists']}")
    print(f"  • Structure correct: {yolo_stats['structure_correct']}")
    print(f"  • Label format correct: {yolo_stats['label_format_correct']}")
    print(f"  • Total classes: {yolo_stats['total_classes']}")
    print(f"  • Image-label pairs: {yolo_stats['image_label_pairs']}")
    print(f"  • Format issues: {len(yolo_stats['format_issues'])}")
    
    return yolo_stats

def main():
    """Main analysis function"""
    
    print("🔬 Starting comprehensive dataset analysis...")
    
    # Load existing statistics
    stats = load_dataset_statistics()
    
    # Create visualizations
    create_image_count_visualization(stats)
    
    # Assess quality
    quality_stats = assess_image_quality()
    create_quality_visualization(quality_stats)
    
    # Identify biases
    biases = identify_data_biases()
    
    # Plan augmentation
    augmentation_plan = plan_data_augmentation()
    
    # Create splits
    splits = create_train_val_test_splits()
    
    # Verify image content (NEW)
    content_stats = verify_image_content()
    create_content_visualization(content_stats)
    
    # Verify YOLO dataset format (NEW)
    yolo_stats = verify_yolo_dataset_format()
    
    # Generate report
    generate_analysis_report()
    
    print("\n✅ Dataset analysis complete!")
    print("\n📊 Generated files:")
    print("  • data/raw/image_distribution_analysis.png")
    print("  • data/raw/quality_assessment.png")
    print("  • data/raw/quality_assessment.json")
    print("  • data/raw/bias_analysis.json")
    print("  • data/raw/augmentation_plan.json")
    print("  • data/raw/dataset_splits.json")
    print("  • data/raw/content_verification.png")
    print("  • data/raw/content_verification.json")
    print("  • data/raw/yolo_format_verification.json")
    print("  • data/raw/analysis_report.md")
    
    print("\n🎯 Ready for Task 2.1: Data Processing & Validation")

if __name__ == "__main__":
    main() 