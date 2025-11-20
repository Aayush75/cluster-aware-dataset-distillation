#!/usr/bin/env python3
"""
Download CIFAR-100 and convert to ImageFolder structure for this repo
"""

import os
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import urllib.request
import tarfile

# Configuration
OUTPUT_DIR = "/home/ssl.distillation/clustering/data/cifar100"
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
TEMP_DIR = "/tmp/cifar100_download"

def download_cifar100():
    """Download CIFAR-100 dataset"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    tar_path = os.path.join(TEMP_DIR, "cifar-100-python.tar.gz")
    
    if not os.path.exists(tar_path):
        print("Downloading CIFAR-100...")
        urllib.request.urlretrieve(CIFAR100_URL, tar_path)
        print("Download complete!")
    
    # Extract
    print("Extracting...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(TEMP_DIR)
    
    return os.path.join(TEMP_DIR, "cifar-100-python")

def unpickle(file):
    """Load pickled CIFAR-100 batch file"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_images_to_folders(data_dict, output_dir, split_name):
    """Convert CIFAR-100 batch to ImageFolder structure"""
    images = data_dict[b'data']
    labels = data_dict[b'fine_labels']
    
    # Reshape images from (N, 3072) to (N, 3, 32, 32)
    images = images.reshape(-1, 3, 32, 32)
    # Transpose to (N, 32, 32, 3) for PIL
    images = images.transpose(0, 2, 3, 1)
    
    print(f"\nProcessing {split_name} split...")
    for idx, (image, label) in enumerate(tqdm(zip(images, labels), total=len(labels))):
        # Create class directory
        class_dir = output_dir / str(label)
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image
        image_path = class_dir / f"{idx:06d}.png"
        img = Image.fromarray(image)
        img.save(image_path)
    
    print(f"Saved {len(labels)} images to {output_dir}")

def main():
    # Download and extract
    cifar_dir = download_cifar100()
    
    # Create output directories
    train_dir = Path(OUTPUT_DIR) / "train"
    val_dir = Path(OUTPUT_DIR) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Process training data
    train_data = unpickle(os.path.join(cifar_dir, "train"))
    save_images_to_folders(train_data, train_dir, "train")
    
    # Process test data (use as validation)
    test_data = unpickle(os.path.join(cifar_dir, "test"))
    save_images_to_folders(test_data, val_dir, "validation")
    
    print(f"\nConversion complete!")
    print(f"Dataset saved to: {OUTPUT_DIR}")
    print(f"Train: {train_dir} (50,000 images)")
    print(f"Val: {val_dir} (10,000 images)")
    print(f"Classes: 100")

if __name__ == "__main__":
    main()
