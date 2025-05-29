# scripts/test_model.py

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import time
import numpy as np
import traceback
from datetime import datetime
import matplotlib.pyplot as plt # For visualization

# --- Helper Function to Add Project Root to Path ---
# (Keep the add_project_root_to_path function as before)
def add_project_root_to_path():
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        print(f"Adding project root to Python path: {project_root}")
        sys.path.insert(0, str(project_root))
    try:
        # Import necessary classes for model definition and potential data loading
        from neural_netwok.model import SignalToMaskUNet
        # We might not need the full Dataset classes if we load directly
        # from neural_netwok.csi_dataset import CsiDataset
        # from neural_netwok.rssi_dataset import RssiDataset
        print("Successfully imported Model class.")
        return True
    except ModuleNotFoundError as e:
        print("\n--- ERROR ---")
        print(f"Could not import required classes: {e}")
        print("Please ensure:")
        print("  1. This script ('test_model.py') is in the 'scripts/' directory.")
        print("  2. The 'neural_netwok/' directory exists in the project root.")
        print("  3. 'model.py' is inside 'neural_netwok/'.")
        print(f"Current sys.path: {sys.path}")
        print("-------------")
        return False
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An unexpected error occurred during import: {e}")
        traceback.print_exc()
        print("-------------")
        return False

# --- Custom Dataset for Loading Specific Files ---
class TestFileDataset(Dataset):
    """Loads data specifically from a list of (feature_path, gt_path) tuples."""
    def __init__(self, sample_paths, feature_dtype=np.float32, gt_dtype=np.float32):
        """
        Args:
            sample_paths (list): List of tuples, where each tuple is (feature_file_path, gt_file_path).
            feature_dtype: Numpy dtype for features.
            gt_dtype: Numpy dtype for ground truth.
        """
        self.sample_paths = sample_paths
        self.feature_dtype = feature_dtype
        self.gt_dtype = gt_dtype
        print(f"Initialized TestFileDataset with {len(sample_paths)} samples.")

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        feature_path, gt_path = self.sample_paths[idx]

        try:
            # Load features
            features = np.load(feature_path).astype(self.feature_dtype)
            # Load ground truth
            ground_truth = np.load(gt_path).astype(self.gt_dtype)

            # Convert to PyTorch Tensors
            feature_tensor = torch.from_numpy(features)

            # Add channel dimension to ground truth if it's 2D
            if ground_truth.ndim == 2:
                ground_truth = np.expand_dims(ground_truth, axis=0) # (H, W) -> (1, H, W)
            ground_truth_tensor = torch.from_numpy(ground_truth)

            return feature_tensor, ground_truth_tensor

        except FileNotFoundError:
            print(f"Error: File not found during loading test sample {idx}. Path: {feature_path} or {gt_path}")
            # Return dummy data or raise error - let's return None and handle in main loop
            return None, None
        except Exception as e:
            print(f"Error loading or processing test sample {idx} ({feature_path}, {gt_path}): {e}")
            traceback.print_exc()
            return None, None

# --- Function to Load Test File Paths ---
def load_test_paths(split_info_path):
    """Loads the list of test file paths from split_info.txt."""
    if not split_info_path.is_file():
        print(f"ERROR: Split info file not found: {split_info_path}")
        return None

    test_paths = []
    try:
        with open(split_info_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("TEST\t"):
                    parts = line.split('\t')
                    if len(parts) == 3:
                        test_paths.append((parts[1], parts[2])) # (feature_path, gt_path)
                    else:
                        print(f"Warning: Malformed TEST line in {split_info_path}: {line}")
        print(f"Loaded {len(test_paths)} test file paths from {split_info_path}")
        return test_paths
    except Exception as e:
        print(f"Error reading split info file {split_info_path}: {e}")
        traceback.print_exc()
        return None

# --- Function to Save Visual Comparison ---
def save_comparison_image(output_dir, index, prediction, ground_truth, filename_prefix="comparison"):
    """Saves an image comparing the prediction and ground truth masks."""
    pred_np = prediction.squeeze().detach().cpu().numpy() # Remove batch/channel dims, move to CPU, convert to numpy
    gt_np = ground_truth.squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'Test Sample {index}')

    im1 = axes[0].imshow(pred_np, cmap='viridis', vmin=0, vmax=1) # Use consistent color map and range
    axes[0].set_title('Predicted Mask')
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(gt_np, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    save_path = output_dir / f"{filename_prefix}_sample_{index}.png"
    try:
        plt.savefig(save_path)
        print(f"Saved comparison image: {save_path}")
    except Exception as e:
        print(f"Error saving comparison image {save_path}: {e}")
    plt.close(fig) # Close the figure to free memory


# --- Main Testing Function ---
def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    training_run_dir = Path(args.training_run_dir)
    if not training_run_dir.is_dir():
        print(f"ERROR: Training run directory not found: {training_run_dir}")
        sys.exit(1)

    model_path = training_run_dir / f"best_model_{args.model_type}.pth"
    if not model_path.is_file():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    split_info_path = training_run_dir / "split_info.txt"
    if not split_info_path.is_file():
        print(f"ERROR: Split info file not found: {split_info_path}")
        sys.exit(1)

    # Create output directory for test results within the training run folder
    test_output_dir = training_run_dir / "test_results"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving test results and visualizations to: {test_output_dir}")

    # --- Load Test Data ---
    test_sample_paths = load_test_paths(split_info_path)
    if test_sample_paths is None or not test_sample_paths:
        print("ERROR: Could not load test file paths or no test paths found.")
        sys.exit(1)

    test_dataset = TestFileDataset(sample_paths=test_sample_paths)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    print(f"Test DataLoader created with {len(test_dataset)} samples.")

    # --- Load Model ---
    if args.model_type == 'csi':
        input_channels = 156
    elif args.model_type == 'rssi':
        input_channels = 3
    else: # Should not happen if argparse choices work
        print(f"ERROR: Invalid model_type '{args.model_type}'")
        sys.exit(1)

    # Need to know base_filters used during training. Add as arg or try to infer?
    # Let's add it as an argument for robustness.
    print(f"Loading model: {model_path}")
    print(f"Instantiating model with input_channels={input_channels}, base_filters={args.base_filters}")
    model = SignalToMaskUNet(input_channels=input_channels, base_filters=args.base_filters).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model weights from {model_path}: {e}")
        traceback.print_exc()
        sys.exit(1)

    model.eval() # Set model to evaluation mode

    # --- Testing Loop ---
    criterion = nn.BCELoss() # Use the same loss function as training
    total_loss = 0.0
    samples_processed = 0
    visualization_count = 0

    print("\n--- Starting Testing ---")
    test_start_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if inputs is None or targets is None:
                print(f"Warning: Skipping batch {i} due to loading error in TestFileDataset.")
                continue

            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0) # Accumulate loss weighted by batch size
            samples_processed += inputs.size(0)

            # Save comparison images for the first few samples/batches
            if visualization_count < args.num_visuals:
                for j in range(inputs.size(0)): # Iterate through samples in the batch
                    if visualization_count < args.num_visuals:
                        sample_idx = i * args.batch_size + j
                        save_comparison_image(test_output_dir, sample_idx,
                                              outputs[j], targets[j],
                                              filename_prefix=f"{args.model_type}_comparison")
                        visualization_count += 1
                    else:
                        break # Stop saving images once limit is reached

            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                 print(f'Batch [{i+1}/{len(test_loader)}] Processed')


    test_end_time = time.time()
    avg_loss = total_loss / samples_processed if samples_processed > 0 else 0
    test_duration = test_end_time - test_start_time

    print("\n--- Testing Finished ---")
    print(f"Total test time: {test_duration:.2f} seconds")
    print(f"Processed {samples_processed} test samples.")
    print(f"Average Test Loss: {avg_loss:.4f}")
    print(f"Saved {visualization_count} comparison images to: {test_output_dir}")
    print("------------------------")

    # Save test results summary
    results_summary_path = test_output_dir / "test_summary.txt"
    with open(results_summary_path, 'w') as f:
        f.write(f"Test Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Tested: {model_path}\n")
        f.write(f"Split Info Used: {split_info_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Samples Tested: {samples_processed}\n")
        f.write(f"Average Test Loss (BCELoss): {avg_loss:.6f}\n")
        f.write(f"Total Test Duration (s): {test_duration:.2f}\n")
        f.write(f"Visualizations Saved: {visualization_count}\n")
    print(f"Test summary saved to: {results_summary_path}")


if __name__ == "__main__":
    if not add_project_root_to_path():
        sys.exit(1)

    # Import after path is set
    from neural_netwok.model import SignalToMaskUNet

    parser = argparse.ArgumentParser(description="Test a trained SignalToMaskUNet model.")

    parser.add_argument('--training_run_dir', type=str, required=True,
                        help="Path to the specific timestamped training output directory (e.g., training_output/csi/20231027_103000)")
    parser.add_argument('--model_type', type=str, required=True, choices=['csi', 'rssi'],
                        help="Type of model being tested ('csi' or 'rssi').")
    # Add base_filters argument as it's needed to instantiate the model correctly
    parser.add_argument('--base_filters', type=int, default=64,
                        help="Base number of filters used when training the model (default: 64). Must match training.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for testing (default: 16)")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of worker processes for DataLoader (default: 4)")
    parser.add_argument('--num_visuals', type=int, default=5,
                        help="Number of comparison images to save (default: 5)")

    args = parser.parse_args()

    main(args)