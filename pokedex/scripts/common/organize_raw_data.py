#!/usr/bin/env python3
"""
Raw Data Organization Script
Organizes Pokemon dataset into proper directory structure
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List

def create_data_summary():
    """Create a comprehensive summary of the raw data organization"""
    
    summary = {
        "dataset_overview": {
            "total_pokemon": 1025,
            "organization": "All Pokemon data under individual ID folders",
            "source": "Mixed: Kaggle dataset (001-151) + Bing search (152-1025)",
            "path": "data/raw/all_pokemon/"
        },
        "directory_structure": {
            "data/raw/all_pokemon/": {
                "0001/": "Bulbasaur - all images (Kaggle + web-scraped)",
                "0002/": "Ivysaur - all images (Kaggle + web-scraped)",
                "...": "...",
                "0151/": "Mew - all images (Kaggle + web-scraped)",
                "0152/": "Chikorita - all images (web-scraped only)",
                "...": "...",
                "1025/": "Last Pokemon - all images (web-scraped only)"
            }
        },
        "data_sources": {
            "pokemon_001_151": {
                "kaggle_dataset": "Original Kaggle dataset images",
                "web_scraped": "Additional Bing search images",
                "total_images": "Variable per Pokemon"
            },
            "pokemon_152_1025": {
                "web_scraped": "Bing search images only",
                "total_images": "Variable per Pokemon"
            }
        },
        "data_quality_notes": [
            "Mixed image quality across all Pokemon",
            "Some folders may contain duplicates",
            "Variable number of images per Pokemon",
            "Different image formats and sizes",
            "May contain irrelevant or low-quality images"
        ],
        "processing_requirements": [
            "Quality filtering for all images",
            "Duplicate removal",
            "Image format standardization",
            "Size normalization",
            "Quality assessment and filtering",
            "Balancing image counts across Pokemon"
        ]
    }
    
    # Save summary to file
    with open("data/raw/dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Dataset summary created: data/raw/dataset_summary.json")
    return summary

def verify_data_organization():
    """Verify that data is properly organized"""
    
    pokemon_path = Path("data/raw/all_pokemon/")
    if not pokemon_path.exists():
        print("❌ Pokemon dataset directory not found")
        return False
    
    pokemon_dirs = [d for d in pokemon_path.iterdir() if d.is_dir()]
    pokemon_dirs.sort()  # Sort by Pokemon ID
    
    print(f"✅ Total Pokemon directories: {len(pokemon_dirs)}")
    
    if len(pokemon_dirs) == 1025:
        print("✅ All 1025 Pokemon directories found!")
        
        # Show some examples
        print("\n📁 Sample Pokemon directories:")
        for i, dir_path in enumerate(pokemon_dirs[:5]):
            print(f"  • {dir_path.name}/")
        
        if len(pokemon_dirs) > 5:
            print(f"  • ... ({len(pokemon_dirs) - 5} more)")
        
        return True
    else:
        print(f"❌ Expected 1025, found {len(pokemon_dirs)}")
        return False

def analyze_image_counts():
    """Analyze image counts per Pokemon"""
    
    pokemon_path = Path("data/raw/all_pokemon/")
    image_counts = {}
    
    for pokemon_dir in pokemon_path.iterdir():
        if pokemon_dir.is_dir():
            pokemon_id = pokemon_dir.name
            image_files = list(pokemon_dir.glob("*.jpg")) + list(pokemon_dir.glob("*.png")) + list(pokemon_dir.glob("*.jpeg"))
            image_counts[pokemon_id] = len(image_files)
    
    # Calculate statistics
    counts = list(image_counts.values())
    total_images = sum(counts)
    avg_images = total_images / len(counts) if counts else 0
    min_images = min(counts) if counts else 0
    max_images = max(counts) if counts else 0
    
    stats = {
        "total_pokemon": len(image_counts),
        "total_images": total_images,
        "average_images_per_pokemon": round(avg_images, 2),
        "min_images_per_pokemon": min_images,
        "max_images_per_pokemon": max_images,
        "image_counts": image_counts
    }
    
    # Save statistics
    with open("data/raw/image_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"📊 Image Statistics:")
    print(f"  • Total Pokemon: {stats['total_pokemon']}")
    print(f"  • Total Images: {stats['total_images']}")
    print(f"  • Average images per Pokemon: {stats['average_images_per_pokemon']}")
    print(f"  • Min images per Pokemon: {stats['min_images_per_pokemon']}")
    print(f"  • Max images per Pokemon: {stats['max_images_per_pokemon']}")
    
    return stats

def create_pokemon_mapping():
    """Create Pokemon ID to name mapping"""
    
    # This would need to be populated with actual Pokemon names
    # For now, create a template
    pokemon_mapping = {}
    
    # Generate mapping for 001-1025
    for i in range(1, 1026):
        pokemon_id = f"{i:04d}"
        pokemon_mapping[pokemon_id] = f"pokemon_{i}"  # Placeholder names
    
    # Save mapping
    with open("data/raw/pokemon_mapping.json", "w") as f:
        json.dump(pokemon_mapping, f, indent=2)
    
    print("✅ Pokemon mapping template created: data/raw/pokemon_mapping.json")
    print("⚠️  Note: Pokemon names need to be populated manually")

def main():
    """Main organization function"""
    
    print("🔧 Organizing Pokemon raw data...")
    
    # Create dataset summary
    summary = create_data_summary()
    
    # Verify organization
    if verify_data_organization():
        print("✅ Data organization verified successfully!")
    else:
        print("❌ Data organization verification failed!")
        return
    
    # Analyze image counts
    stats = analyze_image_counts()
    
    # Create Pokemon mapping
    create_pokemon_mapping()
    
    print("\n📊 Dataset Summary:")
    print(f"  • Total Pokemon: {summary['dataset_overview']['total_pokemon']}")
    print(f"  • Organization: {summary['dataset_overview']['organization']}")
    print(f"  • Source: {summary['dataset_overview']['source']}")
    
    print("\n📁 Directory Structure:")
    print("  data/raw/all_pokemon/")
    print("  ├── 0001/ (Bulbasaur - all images)")
    print("  ├── 0002/ (Ivysaur - all images)")
    print("  ├── ...")
    print("  ├── 0151/ (Mew - all images)")
    print("  ├── 0152/ (Chikorita - web-scraped only)")
    print("  ├── ...")
    print("  └── 1025/ (Last Pokemon - web-scraped only)")
    
    print("\n✅ Raw data organization complete!")

if __name__ == "__main__":
    main() 