# scripts/compare_models.py

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import time
import numpy as np
import pandas as pd
import traceback
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

# --- Helper Function to Add Project Root to Path ---
def add_project_root_to_path():
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        print(f"Adding project root to Python path: {project_root}")
        sys.path.insert(0, str(project_root))
    try:
        from neural_netwok.model import SignalToMaskUNet # Use the correct model class
        print("Successfully imported Model class.")
        return True
    except ModuleNotFoundError as e:
        print("\n--- ERROR ---")
        print(f"Could not import required classes: {e}")
        print("Please ensure:")
        print("  1. This script ('compare_models.py') is in the 'scripts/' directory.")
        print("  2. The 'neural_netwok/' directory exists in the project root.")
        print("  3. 'model.py' (with SignalToMaskUNet) is inside 'neural_netwok/'.")
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
    """Loads features and GT from a list of (feature_path, gt_path) tuples."""
    def __init__(self, sample_paths, feature_shape, feature_dtype=np.float32, gt_dtype=np.float32):
        """
        Args:
            sample_paths (list): List of tuples (feature_file_path, gt_file_path).
            feature_shape (tuple): Expected shape of the feature tensor (e.g., (156, 16, 16)).
            feature_dtype: Numpy dtype for features.
            gt_dtype: Numpy dtype for ground truth.
        """
        self.sample_paths = sample_paths
        self.feature_shape = feature_shape
        self.feature_dtype = feature_dtype
        self.gt_dtype = gt_dtype
        # print(f"Initialized TestFileDataset with {len(sample_paths)} samples.") # Less verbose

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        feature_path, gt_path = self.sample_paths[idx]
        try:
            features = np.load(feature_path).astype(self.feature_dtype)
            ground_truth = np.load(gt_path).astype(self.gt_dtype)

            # Validate shapes
            if features.shape != self.feature_shape:
                raise ValueError(f"Feature shape mismatch. Got {features.shape}, expected {self.feature_shape}")
            # Assuming GT shape is always (360, 360) before adding channel dim
            if ground_truth.shape != (360, 360):
                 # Allow flexibility if GT shape varies slightly, but warn
                 warnings.warn(f"Ground truth shape mismatch for {gt_path}. Got {ground_truth.shape}, expected (360, 360). Check data consistency.")
                 # Attempt to resize if drastically different? Or just proceed? Let's proceed for now.

            feature_tensor = torch.from_numpy(features)
            if ground_truth.ndim == 2:
                ground_truth = np.expand_dims(ground_truth, axis=0) # (H, W) -> (1, H, W)
            ground_truth_tensor = torch.from_numpy(ground_truth)

            # Final check on tensor shapes before returning
            if ground_truth_tensor.shape[1:] != (360, 360): # Check H, W dims
                 warnings.warn(f"Final ground truth tensor shape mismatch for {gt_path}. Got {ground_truth_tensor.shape}, expected (1, 360, 360).")


            return feature_tensor, ground_truth_tensor

        except FileNotFoundError as e:
            print(f"Error: File not found loading sample {idx}. Path: {e.filename}")
            return None, None # Return None tuple to be handled in main loop
        except ValueError as e:
             print(f"Error: Value error (likely shape mismatch) loading sample {idx} ({feature_path}, {gt_path}): {e}")
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
        print(f"Loaded {len(test_paths)} test file paths from {split_info_path}")
        if not test_paths:
             print(f"Warning: No lines starting with 'TEST\\t' found in {split_info_path}")
        return test_paths
    except Exception as e:
        print(f"Error reading split info file {split_info_path}: {e}")
        return None

# --- Function to Save Visual Comparison ---
def save_comparison_image(output_dir, index, prediction, ground_truth, filename_prefix="comparison"):
    """Saves an image comparing the prediction and ground truth masks."""
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output_dir exists
    # Ensure tensors are valid before proceeding
    if prediction is None or ground_truth is None:
        print(f"Warning: Skipping visualization for sample {index} due to missing prediction or ground truth.")
        return

    try:
        pred_np = prediction.squeeze().detach().cpu().numpy()
        gt_np = ground_truth.squeeze().detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'Test Sample {index}')

        im1 = axes[0].imshow(pred_np, cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title('Predicted Mask')
        axes[0].axis('off')
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(gt_np, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = output_dir / f"{filename_prefix}_sample_{index}.png"
        plt.savefig(save_path)
        # print(f"Saved comparison image: {save_path}") # Less verbose during evaluation
        plt.close(fig) # Corrected: close the figure object
    except Exception as e:
        print(f"Error saving comparison image for sample {index} ({save_path}): {e}")
        if 'fig' in locals() and fig is not None:
             plt.close(fig) # Attempt to close if figure exists


# --- Metrics Calculation ---
@torch.no_grad()
def calculate_metrics(predictions, targets, threshold=0.5, epsilon=1e-6):
    """Calculates segmentation metrics for a batch."""
    # Ensure inputs are valid tensors
    if not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
        return {'bce': float('nan'), 'iou': float('nan'), 'dice': float('nan'), 'accuracy': float('nan')}

    batch_size = predictions.size(0)
    metrics = {'bce': 0.0, 'iou': 0.0, 'dice': 0.0, 'accuracy': 0.0}

    try:
        # 1. BCE Loss
        # Ensure target is float for BCE
        bce_loss = nn.functional.binary_cross_entropy(predictions, targets.float(), reduction='none')
        metrics['bce'] = torch.mean(bce_loss).item()

        # 2. Thresholded Metrics
        preds_binary = (predictions >= threshold).float()
        targets_binary = (targets >= 0.5).float() # Ensure GT is also binary 0/1

        # Flatten spatial dimensions: (B, H*W)
        preds_flat = preds_binary.view(batch_size, -1)
        targets_flat = targets_binary.view(batch_size, -1)

        tp = torch.sum(preds_flat * targets_flat, dim=1)
        fp = torch.sum(preds_flat * (1 - targets_flat), dim=1)
        fn = torch.sum((1 - preds_flat) * targets_flat, dim=1)
        tn = torch.sum((1 - preds_flat) * (1 - targets_flat), dim=1)

        # IoU per image
        iou_denominator = tp + fp + fn
        iou = (tp + epsilon) / (iou_denominator + epsilon)
        metrics['iou'] = torch.mean(iou).item()

        # Dice per image
        dice_denominator = 2. * tp + fp + fn
        dice = (2. * tp + epsilon) / (dice_denominator + epsilon)
        metrics['dice'] = torch.mean(dice).item()

        # Accuracy per image
        accuracy_denominator = tp + tn + fp + fn
        accuracy = (tp + tn + epsilon) / (accuracy_denominator + epsilon)
        metrics['accuracy'] = torch.mean(accuracy).item()

    except Exception as e:
        print(f"Error during metric calculation: {e}")
        traceback.print_exc()
        # Return NaN if calculation fails
        return {'bce': float('nan'), 'iou': float('nan'), 'dice': float('nan'), 'accuracy': float('nan')}

    return metrics

# --- Evaluation Function ---
def evaluate_model(model, dataloader, device, threshold, num_visuals=0, visual_output_dir=None, model_name="model"):
    """Evaluates a model on a given dataloader and returns average metrics."""
    model.eval()
    total_metrics = defaultdict(float)
    valid_samples_count = 0
    visualization_count = 0

    print(f"Evaluating {model_name}...")
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            # Handle potential None values from dataset error
            if batch_data is None or batch_data[0] is None or batch_data[1] is None:
                 print(f"Warning: Skipping batch {i} due to loading error in TestFileDataset.")
                 continue

            inputs, targets = batch_data
            # Additional check for tensor type just in case
            if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
                 print(f"Warning: Skipping batch {i} due to non-tensor data.")
                 continue

            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            try:
                outputs = model(inputs)

                # Calculate metrics for this batch
                batch_metrics = calculate_metrics(outputs, targets, threshold)

                # Check if metrics calculation was successful (returned NaNs?)
                if not np.isnan(batch_metrics['bce']): # Check one metric
                    # Accumulate metrics (weighted by batch size for averaging later)
                    for key, value in batch_metrics.items():
                        total_metrics[key] += value * batch_size
                    valid_samples_count += batch_size
                else:
                    print(f"Warning: Skipping metrics accumulation for batch {i} due to calculation error.")


                # Save comparison images if requested
                if visual_output_dir and visualization_count < num_visuals:
                    for j in range(batch_size):
                        if visualization_count < num_visuals:
                            sample_idx_global = i * dataloader.batch_size + j
                            save_comparison_image(visual_output_dir, sample_idx_global,
                                                  outputs[j], targets[j],
                                                  filename_prefix=f"{model_name}_comparison")
                            visualization_count += 1
                        else:
                            break

            except Exception as e:
                 print(f"Error during model inference or visualization for batch {i}: {e}")
                 traceback.print_exc()
                 # Continue to the next batch

            if (i + 1) % 20 == 0 or (i + 1) == len(dataloader):
                 print(f'  Batch [{i+1}/{len(dataloader)}] Processed')

    # Calculate average metrics over valid samples
    avg_metrics = {key: value / valid_samples_count for key, value in total_metrics.items()} if valid_samples_count > 0 else {key: 0.0 for key in total_metrics}
    print(f"Evaluation finished for {model_name}. Processed {valid_samples_count} valid samples.")
    if visual_output_dir:
        print(f"Saved {visualization_count} comparison images to {visual_output_dir}")

    return avg_metrics


# --- Main Comparison Function ---
def main(args):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    csi_run_dir = Path(args.csi_run_dir)
    rssi_run_dir = Path(args.rssi_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Comparison results will be saved to: {output_dir}")

    # Validate input directories and files
    for run_dir, model_type in [(csi_run_dir, 'csi'), (rssi_run_dir, 'rssi')]:
        if not run_dir.is_dir(): print(f"ERROR: {model_type.upper()} run directory not found: {run_dir}"); sys.exit(1)
        model_path = run_dir / f"best_model_{model_type}.pth"
        split_path = run_dir / "split_info.txt"
        if not model_path.is_file(): print(f"ERROR: Model file not found: {model_path}"); sys.exit(1)
        if not split_path.is_file(): print(f"ERROR: Split info file not found: {split_path}"); sys.exit(1)

    # --- Load Test Data ---
    print("\n--- Loading Test Data ---")
    # Load paths using the respective split files
    csi_test_paths = load_test_paths(csi_run_dir / "split_info.txt")
    rssi_test_paths = load_test_paths(rssi_run_dir / "split_info.txt")

    if not csi_test_paths: print("ERROR: Failed to load CSI test paths."); sys.exit(1)
    if not rssi_test_paths: print("ERROR: Failed to load RSSI test paths."); sys.exit(1)

    # Create separate datasets and dataloaders
    csi_test_dataset = TestFileDataset(sample_paths=csi_test_paths, feature_shape=(156, 16, 16))
    rssi_test_dataset = TestFileDataset(sample_paths=rssi_test_paths, feature_shape=(3, 16, 16))

    if len(csi_test_dataset) == 0: print("ERROR: No samples in CSI test dataset."); sys.exit(1)
    if len(rssi_test_dataset) == 0: print("ERROR: No samples in RSSI test dataset."); sys.exit(1)

    csi_test_loader = DataLoader(csi_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    rssi_test_loader = DataLoader(rssi_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("Test DataLoaders created.")

    # --- Load Models ---
    print("\n--- Loading Models ---")
    # Load CSI Model
    csi_model = SignalToMaskUNet(input_channels=156, base_filters=args.csi_base_filters).to(device) # Use specific arg
    csi_model_path = csi_run_dir / "best_model_csi.pth"
    try:
        csi_model.load_state_dict(torch.load(csi_model_path, map_location=device))
        print(f"CSI model weights loaded successfully from {csi_model_path}")
    except Exception as e:
        print(f"ERROR loading CSI model weights: {e}"); traceback.print_exc(); sys.exit(1)

    # Load RSSI Model
    rssi_model = SignalToMaskUNet(input_channels=3, base_filters=args.rssi_base_filters).to(device) # Use specific arg
    rssi_model_path = rssi_run_dir / "best_model_rssi.pth"
    try:
        rssi_model.load_state_dict(torch.load(rssi_model_path, map_location=device))
        print(f"RSSI model weights loaded successfully from {rssi_model_path}")
    except Exception as e:
        print(f"ERROR loading RSSI model weights: {e}"); traceback.print_exc(); sys.exit(1)

    # --- Evaluate Models ---
    print("\n--- Evaluating Models ---")
    threshold = args.threshold

    csi_visual_dir = output_dir / "csi_visuals" if args.num_visuals > 0 else None
    rssi_visual_dir = output_dir / "rssi_visuals" if args.num_visuals > 0 else None

    # Evaluate each model on its respective test loader
    csi_metrics = evaluate_model(csi_model, csi_test_loader, device, threshold,
                                 args.num_visuals, csi_visual_dir, "CSI_Model")
    rssi_metrics = evaluate_model(rssi_model, rssi_test_loader, device, threshold,
                                  args.num_visuals, rssi_visual_dir, "RSSI_Model")

    # --- Report Comparison ---
    print("\n--- Comparison Results ---")

    # Create a pandas DataFrame for nice formatting
    data = {
        'Metric': ['BCE Loss', 'Mean IoU', 'Mean Dice', 'Pixel Accuracy'],
        'CSI Model': [
            csi_metrics.get('bce', float('nan')),
            csi_metrics.get('iou', float('nan')),
            csi_metrics.get('dice', float('nan')),
            csi_metrics.get('accuracy', float('nan'))
        ],
        'RSSI Model': [
            rssi_metrics.get('bce', float('nan')),
            rssi_metrics.get('iou', float('nan')),
            rssi_metrics.get('dice', float('nan')),
            rssi_metrics.get('accuracy', float('nan'))
        ]
    }
    df_results = pd.DataFrame(data)
    df_results.set_index('Metric', inplace=True)

    # Determine winner based on primary metrics (lower BCE, higher IoU/Dice)
    winners = {}
    # Handle potential NaN values during comparison
    if not np.isnan(csi_metrics.get('bce', float('inf'))) and not np.isnan(rssi_metrics.get('bce', float('inf'))):
        if csi_metrics['bce'] < rssi_metrics['bce']: winners['BCE Loss'] = 'CSI'
        elif rssi_metrics['bce'] < csi_metrics['bce']: winners['BCE Loss'] = 'RSSI'
        else: winners['BCE Loss'] = 'Tie'
    else: winners['BCE Loss'] = 'N/A' # If NaN exists

    if not np.isnan(csi_metrics.get('iou', -1)) and not np.isnan(rssi_metrics.get('iou', -1)):
        if csi_metrics['iou'] > rssi_metrics['iou']: winners['Mean IoU'] = 'CSI'
        elif rssi_metrics['iou'] > csi_metrics['iou']: winners['Mean IoU'] = 'RSSI'
        else: winners['Mean IoU'] = 'Tie'
    else: winners['Mean IoU'] = 'N/A'

    if not np.isnan(csi_metrics.get('dice', -1)) and not np.isnan(rssi_metrics.get('dice', -1)):
        if csi_metrics['dice'] > rssi_metrics['dice']: winners['Mean Dice'] = 'CSI'
        elif rssi_metrics['dice'] > csi_metrics['dice']: winners['Mean Dice'] = 'RSSI'
        else: winners['Mean Dice'] = 'Tie'
    else: winners['Mean Dice'] = 'N/A'

    if not np.isnan(csi_metrics.get('accuracy', -1)) and not np.isnan(rssi_metrics.get('accuracy', -1)):
        if csi_metrics['accuracy'] > rssi_metrics['accuracy']: winners['Pixel Accuracy'] = 'CSI'
        elif rssi_metrics['accuracy'] > csi_metrics['accuracy']: winners['Pixel Accuracy'] = 'RSSI'
        else: winners['Pixel Accuracy'] = 'Tie'
    else: winners['Pixel Accuracy'] = 'N/A'


    df_results['Winner'] = df_results.index.map(winners)

    print(df_results.to_string(float_format="%.4f")) # Print formatted table

    # Save results to file
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("--- Model Comparison Summary ---\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CSI Model Run Dir: {csi_run_dir}\n")
        f.write(f"RSSI Model Run Dir: {rssi_run_dir}\n")
        f.write(f"CSI Base Filters: {args.csi_base_filters}\n") # Log specific filters
        f.write(f"RSSI Base Filters: {args.rssi_base_filters}\n") # Log specific filters
        f.write(f"Threshold for IoU/Dice/Acc: {args.threshold}\n")
        f.write(f"Device: {device}\n\n")
        f.write(df_results.to_string(float_format="%.6f")) # Save with more precision
        f.write("\n\nLower BCE is better.\nHigher IoU, Dice, Accuracy are better.\n")

    print(f"\nComparison summary saved to: {summary_path}")
    print("--------------------------")


if __name__ == "__main__":
    # Add project root first
    if not add_project_root_to_path():
        sys.exit(1)

    # Import model class after path is set
    from neural_netwok.model import SignalToMaskUNet # Ensure this matches model.py

    parser = argparse.ArgumentParser(description="Compare trained CSI and RSSI models.")

    parser.add_argument('--csi_run_dir', type=str, required=True,
                        help="Path to the specific timestamped training output directory for the CSI model.")
    parser.add_argument('--rssi_run_dir', type=str, required=True,
                        help="Path to the specific timestamped training output directory for the RSSI model.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save comparison results and visuals.")
    # Separate base filter arguments
    parser.add_argument('--csi_base_filters', type=int, default=64,
                        help="Base filters used for training the CSI model (default: 64). Must match training.")
    parser.add_argument('--rssi_base_filters', type=int, default=64,
                        help="Base filters used for training the RSSI model (default: 64). Must match training.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for evaluation (default: 16)")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help="Threshold for binarizing predictions for IoU/Dice/Accuracy (default: 0.5)")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of worker processes for DataLoader (default: 4)")
    parser.add_argument('--num_visuals', type=int, default=0,
                        help="Number of comparison images to save per model (default: 0 = none)")

    args = parser.parse_args()

    main(args)