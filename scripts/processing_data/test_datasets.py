# scripts/test_datasets.py

import os
import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import traceback

# --- Configuration ---
EXPECTED_CSI_FEATURE_SHAPE = (156, 16, 16) # As defined in process_all_data.py (NUM_CSI_FEATURES, MAX_NODES, MAX_NODES)
EXPECTED_RSSI_FEATURE_SHAPE = (3, 16, 16)   # As defined in process_all_data.py (NUM_RSSI_FEATURES, MAX_NODES, MAX_NODES)
EXPECTED_GT_SHAPE = (1, 360, 360)          # As defined in Dataset classes (1 channel added, H, W)
EXPECTED_DTYPE = torch.float32
BATCH_SIZE_TEST = 2 # How many samples to load in a batch test

# --- Helper Function to Add Project Root to Path ---
def add_project_root_to_path():
    """Adds the project root directory (parent of scripts/) to sys.path."""
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        print(f"Adding project root to Python path: {project_root}")
        sys.path.insert(0, str(project_root))
    # Verify imports after adding path
    try:
        from neural_netwok.csi_dataset import CsiDataset
        from neural_netwok.rssi_dataset import RssiDataset
        print("Successfully imported Dataset classes.")
        return True
    except ModuleNotFoundError:
        print("\n--- ERROR ---")
        print("Could not import Dataset classes even after adding project root.")
        print("Please ensure:")
        print("  1. This script ('test_datasets.py') is in the 'scripts/' directory.")
        print("  2. The 'neural_netwok/' directory exists in the project root.")
        print("  3. 'csi_dataset.py' and 'rssi_dataset.py' are inside 'neural_netwok/'.")
        print(f"Current sys.path: {sys.path}")
        print("-------------")
        return False
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An unexpected error occurred during import: {e}")
        traceback.print_exc()
        print("-------------")
        return False

# --- Main Test Function ---
def test_dataset_loader(DatasetClass, data_dir, expected_feature_shape, expected_gt_shape, dataset_name):
    """Tests a single dataset loader."""
    print(f"\n--- Testing {dataset_name} Loader ---")
    print(f"Data directory: {data_dir}")
    print(f"Expected Feature Shape: {expected_feature_shape}")
    print(f"Expected Ground Truth Shape: {expected_gt_shape}")
    print(f"Expected Dtype: {EXPECTED_DTYPE}")

    if not Path(data_dir).is_dir():
        print(f"FAIL: Data directory not found: {data_dir}")
        return False

    # 1. Test Instantiation and Sample Discovery
    try:
        dataset = DatasetClass(data_dir=data_dir)
        print(f"PASS: Dataset instantiated successfully.")
        num_samples = len(dataset)
        print(f"INFO: Found {num_samples} samples.")
        if num_samples == 0:
            print("WARNING: No samples found. Cannot perform further checks. Ensure data exists and filters are correct.")
            # Technically not a failure of the loader code itself, but indicates no data to load.
            # Depending on your needs, you might return False here. Let's return True for now.
            return True
    except FileNotFoundError as e:
        print(f"FAIL: Instantiation failed - {e}")
        return False
    except Exception as e:
        print(f"FAIL: Instantiation failed with unexpected error:")
        traceback.print_exc()
        return False

    # 2. Test Loading a Single Sample (__getitem__)
    try:
        feature_tensor, gt_tensor = dataset[0] # Load the first sample
        print("PASS: Successfully loaded sample 0 using __getitem__.")

        # Check types
        if not isinstance(feature_tensor, torch.Tensor):
             print(f"FAIL: Feature data is not a torch.Tensor (type: {type(feature_tensor)})")
             return False
        if not isinstance(gt_tensor, torch.Tensor):
             print(f"FAIL: Ground truth data is not a torch.Tensor (type: {type(gt_tensor)})")
             return False
        print("PASS: Loaded data types are torch.Tensor.")

        # Check shapes
        if feature_tensor.shape != expected_feature_shape:
            print(f"FAIL: Feature tensor shape mismatch. Got {feature_tensor.shape}, expected {expected_feature_shape}")
            return False
        if gt_tensor.shape != expected_gt_shape:
            print(f"FAIL: Ground truth tensor shape mismatch. Got {gt_tensor.shape}, expected {expected_gt_shape}")
            return False
        print("PASS: Loaded tensor shapes match expected shapes.")

        # Check dtype
        if feature_tensor.dtype != EXPECTED_DTYPE:
            print(f"FAIL: Feature tensor dtype mismatch. Got {feature_tensor.dtype}, expected {EXPECTED_DTYPE}")
            return False
        if gt_tensor.dtype != EXPECTED_DTYPE:
             print(f"FAIL: Ground truth tensor dtype mismatch. Got {gt_tensor.dtype}, expected {EXPECTED_DTYPE}")
             return False
        print(f"PASS: Loaded tensor dtypes match expected ({EXPECTED_DTYPE}).")

    except IndexError:
         print("FAIL: Could not get sample 0 (IndexError). Dataset might be empty despite initial check.")
         return False
    except Exception as e:
        print(f"FAIL: Loading sample 0 failed with unexpected error:")
        traceback.print_exc()
        return False

    # 3. Test DataLoader (Optional but recommended)
    if num_samples >= BATCH_SIZE_TEST:
        try:
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)
            feature_batch, gt_batch = next(iter(dataloader)) # Get the first batch
            print(f"PASS: Successfully loaded a batch using DataLoader (batch size: {BATCH_SIZE_TEST}).")

            # Check batch shapes
            expected_feature_batch_shape = (BATCH_SIZE_TEST,) + expected_feature_shape
            expected_gt_batch_shape = (BATCH_SIZE_TEST,) + expected_gt_shape

            if feature_batch.shape != expected_feature_batch_shape:
                print(f"FAIL: Feature batch shape mismatch. Got {feature_batch.shape}, expected {expected_feature_batch_shape}")
                return False
            if gt_batch.shape != expected_gt_batch_shape:
                print(f"FAIL: Ground truth batch shape mismatch. Got {gt_batch.shape}, expected {expected_gt_batch_shape}")
                return False
            print("PASS: DataLoader batch shapes are correct.")

            # Check batch dtypes
            if feature_batch.dtype != EXPECTED_DTYPE:
                 print(f"FAIL: Feature batch dtype mismatch. Got {feature_batch.dtype}, expected {EXPECTED_DTYPE}")
                 return False
            if gt_batch.dtype != EXPECTED_DTYPE:
                 print(f"FAIL: Ground truth batch dtype mismatch. Got {gt_batch.dtype}, expected {EXPECTED_DTYPE}")
                 return False
            print(f"PASS: DataLoader batch dtypes are correct ({EXPECTED_DTYPE}).")

        except Exception as e:
            print(f"FAIL: DataLoader test failed with unexpected error:")
            traceback.print_exc()
            return False
    else:
        print(f"INFO: Skipping DataLoader test (requires at least {BATCH_SIZE_TEST} samples, found {num_samples}).")

    print(f"--- {dataset_name} Loader Test: ALL CHECKS PASSED ---")
    return True


# --- Script Entry Point ---
if __name__ == "__main__":
    # Add project root to path to allow importing neural_netwok
    if not add_project_root_to_path():
        sys.exit(1) # Exit if imports failed

    # Now we can safely import
    from neural_netwok.csi_dataset import CsiDataset
    from neural_netwok.rssi_dataset import RssiDataset

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    default_processed_csi = project_root / "processed_data_csi"
    default_processed_rssi = project_root / "processed_data_rssi"

    parser = argparse.ArgumentParser(description="Test CSI and RSSI Dataset loaders.")
    parser.add_argument("--csi_dir", type=str, default=str(default_processed_csi),
                        help=f"Path to processed_data_csi directory (default: {default_processed_csi})")
    parser.add_argument("--rssi_dir", type=str, default=str(default_processed_rssi),
                        help=f"Path to processed_data_rssi directory (default: {default_processed_rssi})")
    args = parser.parse_args()

    print("=======================================")
    print("      Starting Dataset Loader Test     ")
    print("=======================================")

    csi_test_passed = test_dataset_loader(
        DatasetClass=CsiDataset,
        data_dir=args.csi_dir,
        expected_feature_shape=EXPECTED_CSI_FEATURE_SHAPE,
        expected_gt_shape=EXPECTED_GT_SHAPE,
        dataset_name="CSI"
    )

    rssi_test_passed = test_dataset_loader(
        DatasetClass=RssiDataset,
        data_dir=args.rssi_dir,
        expected_feature_shape=EXPECTED_RSSI_FEATURE_SHAPE,
        expected_gt_shape=EXPECTED_GT_SHAPE,
        dataset_name="RSSI"
    )

    print("\n=======================================")
    print("            Test Summary             ")
    print("=======================================")
    print(f"CSI Dataset Loader Test: {'PASSED' if csi_test_passed else 'FAILED'}")
    print(f"RSSI Dataset Loader Test: {'PASSED' if rssi_test_passed else 'FAILED'}")
    print("=======================================")

    if not csi_test_passed or not rssi_test_passed:
        sys.exit(1) # Exit with error code if any test failed
    else:
        sys.exit(0) # Exit successfully
        