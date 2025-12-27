"""
Custom PyTorch Dataset for ImageNet-1K stored in parquet files.
Reads images directly from parquet without extracting to disk.
"""

import os
import csv
import io
from PIL import Image
import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
from pathlib import Path


class ImageNetParquetDataset(Dataset):
    """
    Dataset that loads images directly from parquet files using CSV mappings.
    
    Args:
        parquet_dir: Directory containing parquet files (e.g., train-xxxxx.parquet)
        csv_path: Path to CSV with columns: image_path, pseudo_label_class_index (or true_label_index)
        label_column: Which column to use as label ('pseudo_label_class_index' or 'true_label_index')
        transform: Optional transform to apply to images
        cache_parquet: Whether to cache loaded parquet files in memory (uses lots of RAM!)
    """
    
    def __init__(self, parquet_dir, csv_path, label_column='pseudo_label_class_index', 
                 transform=None, cache_parquet=False):
        self.parquet_dir = parquet_dir
        self.transform = transform
        self.cache_parquet = cache_parquet
        self.label_column = label_column
        
        # Load CSV mappings
        print(f"Loading CSV from: {csv_path}")
        self.samples = []
        self.targets = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = row['image_path']
                label = int(row[label_column])
                self.samples.append(img_path)
                self.targets.append(label)
        
        print(f"Loaded {len(self.samples)} samples from CSV")
        
        # Parse image paths to extract parquet file and index
        # Format expected: "train-00001-of-00294/12345" or similar
        self.parquet_cache = {}
        self._parse_image_paths()
    
    def _parse_image_paths(self):
        """Parse image paths to extract parquet file names and indices."""
        self.image_locations = []  # List of (parquet_file, image_index)
        
        # Determine dataset split from first sample
        first_path = self.samples[0]
        if first_path.startswith('train_'):
            split = 'train'
            total_files = 294
        elif first_path.startswith('test_'):
            split = 'test'
            total_files = 28
        elif first_path.startswith('validation_'):
            split = 'validation'
            total_files = 14
        else:
            # Fallback: try to detect from path
            split = 'train'
            total_files = 294
            print(f"Warning: Could not detect split from path '{first_path}', assuming train")
        
        for img_path in self.samples:
            # Parse format: "train_126_210" -> parquet_num=126, row_idx=210
            # Format: <split>_<parquet_num>_<row_idx>
            parts = img_path.split('_')
            
            if len(parts) != 3:
                raise ValueError(f"Unexpected image_path format: {img_path}. Expected '<split>_<parquet_num>_<row_idx>'")
            
            parquet_num = int(parts[1])
            img_idx = int(parts[2])
            
            # Construct parquet filename: train-00126-of-00294.parquet
            parquet_name = f"{split}-{parquet_num:05d}-of-{total_files:05d}.parquet"
            parquet_file = os.path.join(self.parquet_dir, parquet_name)
            
            self.image_locations.append((parquet_file, img_idx))
        
        print(f"Parsed {len(self.image_locations)} image locations")
        
        # Get unique parquet files
        unique_parquets = set(loc[0] for loc in self.image_locations)
        print(f"Dataset spans {len(unique_parquets)} parquet files from {split} split")
    
    def _load_parquet(self, parquet_file):
        """Load a parquet file, with optional caching."""
        if self.cache_parquet and parquet_file in self.parquet_cache:
            return self.parquet_cache[parquet_file]
        
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
        
        if self.cache_parquet:
            self.parquet_cache[parquet_file] = df
        
        return df
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Get image and label at index.
        
        Returns:
            tuple: (image, label) where image is a PIL Image
        """
        parquet_file, img_idx = self.image_locations[index]
        label = self.targets[index]
        
        # Load parquet file
        df = self._load_parquet(parquet_file)
        
        # Get image data at the specific index
        img_data = df.iloc[img_idx]['image']
        
        # Convert to PIL Image
        if isinstance(img_data, dict):
            if 'bytes' in img_data:
                img_bytes = img_data['bytes']
            elif 'path' in img_data:
                with open(img_data['path'], 'rb') as f:
                    img_bytes = f.read()
            else:
                raise ValueError(f"Unknown image data format: {img_data}")
        elif isinstance(img_data, bytes):
            img_bytes = img_data
        else:
            img_bytes = bytes(img_data)
        
        image = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


# Wrapper to make it compatible with the FKD training code
class ImageNetParquetDatasetFKD(ImageNetParquetDataset):
    """
    Extended version compatible with FKD training (returns dummy values for FKD-specific features).
    """
    
    def __init__(self, parquet_dir, csv_path, label_column='pseudo_label_class_index',
                 transform=None, cache_parquet=False, weights_map=None, mode='fkd_save',
                 args_epoch=None, args_bs=None):
        super().__init__(parquet_dir, csv_path, label_column, transform, cache_parquet)
        self.weights_map = weights_map if weights_map else {}
        self.mode = mode
        self.last_mix_config = None
        self.epoch = None
    
    def set_epoch(self, epoch):
        """Dummy method for compatibility with FKD training."""
        self.epoch = epoch
    
    def __getitem__(self, index):
        """
        Get item compatible with FKD training format.
        
        Returns:
            tuple: (image, label, flip_status, coords_status, weight)
        """
        image, label = super().__getitem__(index)
        
        # Get weight (default 1.0)
        img_path = self.samples[index]
        weight = self.weights_map.get(img_path, 1.0)
        
        # Return dummy values for FKD compatibility
        flip_status = None
        coords_status = torch.zeros(4)  # dummy coords
        
        return image, label, flip_status, coords_status, weight


def create_imagenet_parquet_dataset(parquet_dir, csv_path, label_column='pseudo_label_class_index',
                                     transform=None, use_fkd=True, **kwargs):
    """
    Factory function to create the appropriate dataset.
    
    Args:
        parquet_dir: Directory containing parquet files
        csv_path: Path to CSV with image paths and labels
        label_column: Column name for labels
        transform: Transforms to apply
        use_fkd: Whether to use FKD-compatible version
        **kwargs: Additional arguments for FKD version
    
    Returns:
        Dataset instance
    """
    if use_fkd:
        return ImageNetParquetDatasetFKD(
            parquet_dir=parquet_dir,
            csv_path=csv_path,
            label_column=label_column,
            transform=transform,
            **kwargs
        )
    else:
        return ImageNetParquetDataset(
            parquet_dir=parquet_dir,
            csv_path=csv_path,
            label_column=label_column,
            transform=transform
        )
