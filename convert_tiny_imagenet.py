#!/usr/bin/env python3
"""
Convert Hugging Face Tiny-ImageNet parquet dataset to ImageFolder structure
"""

import os
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Configuration
HF_DATASET = "zh-plus/tiny-imagenet"
OUTPUT_DIR = "/home/ssl.distillation/WMDD/datasets/tiny-imagenet"

print(f"Loading dataset from Hugging Face: {HF_DATASET}")
dataset = load_dataset(HF_DATASET)

# Create output directories
train_dir = Path(OUTPUT_DIR) / "train"
val_dir = Path(OUTPUT_DIR) / "val"
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

def save_split(split_name, output_dir):
    """Save a dataset split to ImageFolder structure"""
    split_data = dataset[split_name]
    
    print(f"\nProcessing {split_name} split...")
    for idx, example in enumerate(tqdm(split_data)):
        # Get image and label
        image = example['image']
        label = example['label']
        
        # Create class directory
        class_dir = output_dir / str(label)
        class_dir.mkdir(exist_ok=True)
        
        # Save image
        image_path = class_dir / f"{idx:06d}.JPEG"
        if isinstance(image, Image.Image):
            image.save(image_path)
        else:
            # Handle if image is already a PIL image or needs conversion
            Image.fromarray(image).save(image_path)
    
    print(f"Saved {len(split_data)} images to {output_dir}")

# Process train and validation splits
save_split("train", train_dir)
save_split("valid", val_dir)

print(f"\nConversion complete!")
print(f"Dataset saved to: {OUTPUT_DIR}")
print(f"Train: {train_dir}")
print(f"Val: {val_dir}")
