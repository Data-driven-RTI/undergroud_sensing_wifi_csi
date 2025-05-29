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
import glob


def add_project_root_to_path():
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from neural_netwok.model import SignalToMaskUNet, AdaptedFCN, AdaptedDeepLabV3
        return True
    except ModuleNotFoundError as e:
        print("\n--- ERROR ---")
        print(f"Could not import required classes: {e}")
        print("Please ensure:")
        print("  1. This script ('compare_all_models.py') is in the 'scripts/' directory.")
        print("  2. The 'neural_netwok/' directory exists in the project root.")
        print("  3. 'model.py' (with all model definitions) is inside 'neural_netwok/'.")
        print(f"Current sys.path: {sys.path}")
        print("-------------")
        return False
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An unexpected error occurred during import: {e}")
        traceback.print_exc()
        print("-------------")
        return False


class TestFileDataset(Dataset):
    def __init__(self, sample_paths, feature_shape, feature_dtype=np.float32, gt_dtype=np.float32):
        self.sample_paths = sample_paths
        self.feature_shape = feature_shape
        self.feature_dtype = feature_dtype
        self.gt_dtype = gt_dtype

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        feature_path, gt_path = self.sample_paths[idx]
        try:
            features = np.load(feature_path).astype(self.feature_dtype)
            ground_truth = np.load(gt_path).astype(self.gt_dtype)
            if features.shape != self.feature_shape:
                raise ValueError(f"Feature shape mismatch. Got {features.shape}, expected {self.feature_shape}")
            if ground_truth.shape != (360, 360):
                 warnings.warn(f"Ground truth shape mismatch for {gt_path}. Got {ground_truth.shape}, expected (360, 360). Check data consistency.")
            feature_tensor = torch.from_numpy(features)
            if ground_truth.ndim == 2: ground_truth = np.expand_dims(ground_truth, axis=0)
            ground_truth_tensor = torch.from_numpy(ground_truth)
            if ground_truth_tensor.shape[1:] != (360, 360):
                 warnings.warn(f"Final ground truth tensor shape mismatch for {gt_path}. Got {ground_truth_tensor.shape}, expected (1, 360, 360).")
            return feature_tensor, ground_truth_tensor
        except FileNotFoundError as e: print(f"Error: File not found loading sample {idx}. Path: {e.filename}"); return None, None
        except ValueError as e: print(f"Error: Value error loading sample {idx} ({feature_path}, {gt_path}): {e}"); return None, None
        except Exception as e: print(f"Error loading or processing test sample {idx} ({feature_path}, {gt_path}): {e}"); traceback.print_exc(); return None, None


def load_test_paths(split_info_path):
    if not split_info_path.is_file(): print(f"ERROR: Split info file not found: {split_info_path}"); return None
    test_paths = []
    try:
        with open(split_info_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("TEST\t"):
                    parts = line.split('\t')
                    if len(parts) == 3: test_paths.append((parts[1], parts[2]))
        print(f"Loaded {len(test_paths)} test file paths from {split_info_path}")
        if not test_paths: print(f"Warning: No lines starting with 'TEST\\t' found in {split_info_path}")
        return test_paths
    except Exception as e: print(f"Error reading split info file {split_info_path}: {e}"); return None


def save_comparison_image(output_dir, index, prediction, ground_truth, filename_prefix="comparison"):
     output_dir.mkdir(parents=True, exist_ok=True)
     if prediction is None or ground_truth is None: print(f"Warning: Skipping visualization for sample {index} due to missing data."); return
     try:
        pred_np = prediction.squeeze().detach().cpu().numpy()
        gt_np = ground_truth.squeeze().detach().cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5)); fig.suptitle(f'Test Sample {index}')
        im1 = axes[0].imshow(pred_np, cmap='viridis', vmin=0, vmax=1); axes[0].set_title('Predicted Mask'); axes[0].axis('off'); fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        im2 = axes[1].imshow(gt_np, cmap='viridis', vmin=0, vmax=1); axes[1].set_title('Ground Truth Mask'); axes[1].axis('off'); fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); save_path = output_dir / f"{filename_prefix}_sample_{index}.png"; plt.savefig(save_path); plt.close(fig)
     except Exception as e: print(f"Error saving comparison image for sample {index}: {e}"); plt.close(fig)


@torch.no_grad()
def calculate_metrics(predictions, targets, threshold=0.5, epsilon=1e-6):
    if not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor): return {'bce': float('nan'), 'iou': float('nan'), 'dice': float('nan'), 'accuracy': float('nan')}
    batch_size = predictions.size(0); metrics = {'bce': 0.0, 'iou': 0.0, 'dice': 0.0, 'accuracy': 0.0}
    try:
        bce_loss = nn.functional.binary_cross_entropy(predictions, targets.float(), reduction='mean'); metrics['bce'] = bce_loss.item() # Use mean reduction directly
        preds_binary = (predictions >= threshold).float(); targets_binary = (targets >= 0.5).float()
        preds_flat = preds_binary.view(batch_size, -1); targets_flat = targets_binary.view(batch_size, -1)
        tp = torch.sum(preds_flat * targets_flat, dim=1); fp = torch.sum(preds_flat * (1 - targets_flat), dim=1); fn = torch.sum((1 - preds_flat) * targets_flat, dim=1); tn = torch.sum((1 - preds_flat) * (1 - targets_flat), dim=1)
        iou = (tp + epsilon) / (tp + fp + fn + epsilon); metrics['iou'] = torch.mean(iou).item()
        dice = (2. * tp + epsilon) / (2. * tp + fp + fn + epsilon); metrics['dice'] = torch.mean(dice).item()
        accuracy = (tp + tn + epsilon) / (tp + tn + fp + fn + epsilon); metrics['accuracy'] = torch.mean(accuracy).item()
    except Exception as e: print(f"Error during metric calculation: {e}"); traceback.print_exc(); return {'bce': float('nan'), 'iou': float('nan'), 'dice': float('nan'), 'accuracy': float('nan')}
    return metrics


def evaluate_model(model, dataloader, device, threshold, num_visuals=0, visual_output_dir=None, model_name="model"):
    model.eval(); total_metrics = defaultdict(float); valid_samples_count = 0; visualization_count = 0
    print(f"Evaluating {model_name}...")
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            if batch_data is None or batch_data[0] is None or batch_data[1] is None: print(f"Warning: Skipping batch {i} due to loading error."); continue
            inputs, targets = batch_data
            if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor): print(f"Warning: Skipping batch {i} due to non-tensor data."); continue
            inputs, targets = inputs.to(device), targets.to(device); batch_size = inputs.size(0)
            try:
                outputs = model(inputs)
                batch_metrics = calculate_metrics(outputs, targets, threshold)
                if not np.isnan(batch_metrics['bce']):
                    for key, value in batch_metrics.items():
                        total_metrics[key] += value * batch_size
                    valid_samples_count += batch_size
                else: print(f"Warning: Skipping metrics accumulation for batch {i} due to calculation error.")
                if visual_output_dir and visualization_count < num_visuals:
                    for j in range(batch_size):
                        if visualization_count < num_visuals:
                            sample_idx_global = i * dataloader.batch_size + j; save_comparison_image(visual_output_dir, sample_idx_global, outputs[j], targets[j], filename_prefix=f"{model_name}_comparison"); visualization_count += 1
                        else: break
            except Exception as e: print(f"Error during model inference or visualization for batch {i}: {e}"); traceback.print_exc()
            if (i + 1) % 50 == 0 or (i + 1) == len(dataloader): print(f'  Batch [{i+1}/{len(dataloader)}] Processed')

    avg_metrics = {key: value / valid_samples_count for key, value in total_metrics.items()} if valid_samples_count > 0 else {key: float('nan') for key in total_metrics}
    print(f"Evaluation finished for {model_name}. Processed {valid_samples_count} valid samples.")
    if visual_output_dir: print(f"Saved {visualization_count} comparison images to {visual_output_dir}")
    return avg_metrics


def find_completed_runs(base_run_dir):
    base_run_dir = Path(base_run_dir)
    completed_runs = []
    if not base_run_dir.is_dir():
        print(f"ERROR: Base run directory not found: {base_run_dir}")
        return completed_runs

    print(f"Scanning for completed runs in: {base_run_dir}")
    for data_type_dir in base_run_dir.iterdir():
        if not data_type_dir.is_dir() or data_type_dir.name not in ['csi', 'rssi']:
            continue
        data_type = data_type_dir.name
        for arch_dir in data_type_dir.iterdir():
            if not arch_dir.is_dir() or arch_dir.name not in ['unet', 'fcn', 'deeplab']:
                continue
            architecture = arch_dir.name
            for timestamp_dir in arch_dir.iterdir():
                if not timestamp_dir.is_dir():
                    continue
                timestamp = timestamp_dir.name
                split_file = timestamp_dir / "split_info.txt"
                model_files = list(timestamp_dir.glob(f"best_model_{data_type}_{architecture}.pth"))
                if not model_files:
                    model_files = list(timestamp_dir.glob(f"best_model_{data_type}.pth"))

                if split_file.is_file() and model_files:
                    model_file = model_files[0]
                    run_info = {
                        "data_type": data_type,
                        "architecture": architecture,
                        "timestamp": timestamp,
                        "run_path": timestamp_dir,
                        "model_path": model_file,
                        "split_path": split_file
                    }
                    completed_runs.append(run_info)
                    print(f"  Found completed run: {data_type}/{architecture}/{timestamp}")
                else:
                     print(f"  Skipping incomplete run: {data_type}/{architecture}/{timestamp} (Missing model or split file)")

    print(f"Found {len(completed_runs)} completed runs in total.")
    return completed_runs


def main(args):
    from neural_netwok.model import AdaptedDeepLabV3, AdaptedFCN, SignalToMaskUNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_run_dir = Path(args.base_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation results will be saved to: {output_dir}")

    completed_runs = find_completed_runs(base_run_dir)
    if not completed_runs:
        print("No completed runs found. Exiting.")
        sys.exit(1)

    all_results = []
    base_filters = args.base_filters

    print("\n--- Evaluating All Completed Models ---")
    for run_info in completed_runs:
        data_type = run_info["data_type"]
        architecture = run_info["architecture"]
        run_path = run_info["run_path"]
        model_path = run_info["model_path"]
        split_path = run_info["split_path"]
        model_name = f"{data_type.upper()}_{architecture.upper()}" # e.g., CSI_UNET

        print(f"\n--- Processing: {model_name} (Path: {run_path}) ---")

        test_paths = load_test_paths(split_path)
        if not test_paths:
            print(f"ERROR: Failed to load test paths for {model_name}. Skipping evaluation.")
            continue

        if data_type == 'csi':
            feature_shape = (156, 16, 16)
            input_channels = 156
        else: # rssi
            feature_shape = (3, 16, 16)
            input_channels = 3

        test_dataset = TestFileDataset(sample_paths=test_paths, feature_shape=feature_shape)
        if len(test_dataset) == 0:
            print(f"ERROR: Test dataset is empty for {model_name}. Skipping evaluation.")
            continue

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print(f"Test DataLoader created for {model_name} with {len(test_dataset)} samples.")


        print(f"Loading model: {architecture} with base_filters={base_filters}")
        try:
            if architecture == 'unet':
                model = SignalToMaskUNet(input_channels=input_channels, base_filters=base_filters).to(device)
            elif architecture == 'fcn':
                model = AdaptedFCN(input_channels=input_channels, base_filters=base_filters).to(device)
            elif architecture == 'deeplab':
                model = AdaptedDeepLabV3(input_channels=input_channels, base_filters=base_filters).to(device)
            else:
                print(f"ERROR: Unknown architecture '{architecture}' found for run {run_path}. Skipping.")
                continue

            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model weights loaded successfully from {model_path}")

        except Exception as e:
            print(f"ERROR loading model or weights for {model_name}: {e}")
            traceback.print_exc()
            continue

        visual_dir = output_dir / f"{model_name}_visuals" if args.num_visuals > 0 else None
        metrics = evaluate_model(model, test_loader, device, args.threshold,
                                 args.num_visuals, visual_dir, model_name)

        result_entry = {
            "Model": model_name,
            "Data Type": data_type.upper(),
            "Architecture": architecture.upper(),
            "Run Path": str(run_path),
            **metrics
        }
        all_results.append(result_entry)


    print("\n--- Overall Comparison Results ---")

    if not all_results:
        print("No models were successfully evaluated.")
        sys.exit(0)

    df_results = pd.DataFrame(all_results)

    metric_cols = ['bce', 'iou', 'dice', 'accuracy']
    id_cols = ['Model', 'Data Type', 'Architecture']
    other_cols = ['Run Path']
    all_cols_ordered = id_cols + metric_cols + other_cols
    for col in all_cols_ordered:
        if col not in df_results.columns:
            df_results[col] = float('nan')
    df_results = df_results[all_cols_ordered]

    try:
        df_pivot = df_results.pivot_table(index=['Data Type', 'Architecture'],
                                          values=metric_cols)
        print("\n--- Pivoted Results Table ---")
        print(df_pivot.to_string(float_format="%.4f"))
    except Exception as e:
        print("\nCould not create pivoted table (maybe duplicate runs or other issue):", e)
        df_pivot = None


    print("\n--- Full Results Table ---")
    print(df_results.to_string(index=False, float_format="%.6f"))

    summary_path_txt = output_dir / "comparison_summary_all.txt"
    summary_path_csv = output_dir / "comparison_summary_all.csv"

    with open(summary_path_txt, 'w') as f:
        f.write("--- All Model Comparison Summary ---\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Base Run Dir: {base_run_dir}\n")
        f.write(f"Base Filters Used for Loading: {args.base_filters}\n")
        f.write(f"Threshold for Metrics: {args.threshold}\n")
        f.write(f"Device: {device}\n\n")
        if df_pivot is not None:
            f.write("--- Pivoted Results ---\n")
            f.write(df_pivot.to_string(float_format="%.6f"))
            f.write("\n\n")
        f.write("--- Full Results ---\n")
        f.write(df_results.to_string(index=False, float_format="%.6f"))
        f.write("\n\nLower BCE is better.\nHigher IoU, Dice, Accuracy are better.\n")

    try:
        df_results.to_csv(summary_path_csv, index=False, float_format="%.8f")
        print(f"\nComparison summary saved to: {summary_path_txt}")
        print(f"Comparison summary saved to CSV: {summary_path_csv}")
    except Exception as e:
         print(f"Error saving CSV summary: {e}")


    print("--------------------------")


if __name__ == "__main__":
    if not add_project_root_to_path():
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Compare ALL trained models found in a base directory.")

    parser.add_argument('--base_run_dir', type=str, required=True,
                        help="Path to the base directory containing the parallel training run outputs (e.g., ./training_runs_poster).")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the overall comparison results.")
    parser.add_argument('--base_filters', type=int, default=64,
                        help="Base filters used for ALL training runs (default: 64). MUST match the value used during training.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size for evaluation (default: 16)")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help="Threshold for binarizing predictions for IoU/Dice/Accuracy (default: 0.5)")
    parser.add_argument('--num_workers', type=int, default=0, # Default to 0 for evaluation, often safer
                        help="Number of worker processes for DataLoader during evaluation (default: 0)")
    parser.add_argument('--num_visuals', type=int, default=0,
                        help="Number of comparison images to save per model (default: 0 = none)")

    args = parser.parse_args()

    main(args)