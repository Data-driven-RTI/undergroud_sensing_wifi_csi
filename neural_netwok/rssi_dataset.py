import os
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import random
import traceback

class RssiDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None, include_datasets=None, exclude_datasets=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []

        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Processed data directory not found: {self.data_dir}")

        print(f"Scanning for samples in: {self.data_dir}")
        for dataset_path in self.data_dir.iterdir():
            if not dataset_path.is_dir():
                continue

            dataset_name = dataset_path.name
            if include_datasets and dataset_name not in include_datasets:
                print(f"  Skipping dataset (not included): {dataset_name}")
                continue
            if exclude_datasets and dataset_name in exclude_datasets:
                print(f"  Skipping dataset (excluded): {dataset_name}")
                continue

            print(f"  Processing dataset: {dataset_name}")
            for timestamp_path in dataset_path.iterdir():
                if timestamp_path.is_dir():
                    feature_file = timestamp_path / "rssi_features.npy"
                    gt_file = timestamp_path / "ground_truth.npy"

                    if feature_file.is_file() and gt_file.is_file():
                        self.samples.append((str(feature_file), str(gt_file)))
                    else:
                        print(f"    Warning: Missing features or GT file in {timestamp_path}. Skipping.")

        if not self.samples:
            print(f"Warning: No valid samples found in {self.data_dir} with current filters.")
        else:
             print(f"Found {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError("Index out of bounds")

        feature_path, gt_path = self.samples[idx]

        try:
            features = np.load(feature_path).astype(np.float32)
            ground_truth = np.load(gt_path).astype(np.float32)

            feature_tensor = torch.from_numpy(features)

            if ground_truth.ndim == 2:
                ground_truth = np.expand_dims(ground_truth, axis=0)
            ground_truth_tensor = torch.from_numpy(ground_truth)

            if self.transform:
                feature_tensor = self.transform(feature_tensor)
            if self.target_transform:
                ground_truth_tensor = self.target_transform(ground_truth_tensor)

            return feature_tensor, ground_truth_tensor

        except FileNotFoundError:
            print(f"Error: File not found loading sample {idx}. Path: {feature_path} or {gt_path}")
            print("Attempting to load next sample...")
            return self.__getitem__((idx + 1) % len(self.samples)) if len(self.samples) > 0 else (None, None)
        except Exception as e:
            print(f"Error loading or processing sample {idx} ({feature_path}, {gt_path}): {e}")
            traceback.print_exc()
            print("Attempting to load next sample...")
            return self.__getitem__((idx + 1) % len(self.samples)) if len(self.samples) > 0 else (None, None)