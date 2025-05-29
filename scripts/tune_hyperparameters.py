# scripts/tune_hyperparameters.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
import time
import random
import numpy as np
import traceback
from datetime import datetime
import optuna # Import Optuna
from optuna.trial import TrialState # For pruning status

# --- Helper Function to Add Project Root to Path ---
# (Same as in train_model.py)
def add_project_root_to_path():
    script_dir = Path(__file__).parent.resolve(); project_root = script_dir.parent
    if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))
    try:
        from neural_netwok.csi_dataset import CsiDataset; from neural_netwok.rssi_dataset import RssiDataset
        from neural_netwok.model import SignalToMaskUNet, DiceLoss
        print("Successfully imported Dataset, Model, and Loss classes for tuning.")
        return True
    except Exception as e: print(f"\n--- ERROR: Import failed: {e} ---"); return False

# --- Set Random Seed ---
# (Same as in train_model.py)
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# --- Training & Validation Functions (Copied/Adapted from train_model.py) ---
# NOTE: These are simplified for tuning - less verbose printing
def train_one_epoch_tuned(model, dataloader, criterion_bce, criterion_dice, bce_weight, dice_weight, optimizer, device):
    model.train(); running_loss = 0.0
    for batch_data in dataloader:
        if batch_data is None or batch_data[0] is None or batch_data[1] is None: continue
        inputs, targets = batch_data; inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(); outputs = model(inputs)
        loss_bce = criterion_bce(outputs, targets.float()); loss_dice = criterion_dice(outputs, targets)
        loss = (bce_weight * loss_bce) + (dice_weight * loss_dice)
        loss.backward(); optimizer.step(); running_loss += loss.item()
    return running_loss / len(dataloader) if len(dataloader) > 0 else 0

def validate_tuned(model, dataloader, criterion_bce, criterion_dice, bce_weight, dice_weight, device):
    model.eval(); running_loss = 0.0
    with torch.no_grad():
        for batch_data in dataloader:
            if batch_data is None or batch_data[0] is None or batch_data[1] is None: continue
            inputs, targets = batch_data; inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss_bce = criterion_bce(outputs, targets.float()); loss_dice = criterion_dice(outputs, targets)
            loss = (bce_weight * loss_bce) + (dice_weight * loss_dice)
            running_loss += loss.item()
    return running_loss / len(dataloader) if len(dataloader) > 0 else float('inf') # Return inf if no batches

# --- Optuna Objective Function ---
def objective(trial: optuna.trial.Trial, args):
    """Defines a single training trial for Optuna."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed) # Use fixed seed for dataset split, but allow HPs to vary

    # --- Suggest Hyperparameters ---
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    # Only suggest weight decay if using AdamW
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True) if args.optimizer == 'adamw' else 0.0
    # Suggest loss weights (ensure they sum close to 1, e.g., sample one and derive other)
    # dice_weight = trial.suggest_float("dice_weight", 0.1, 0.9)
    # bce_weight = 1.0 - dice_weight
    # Simpler: suggest ratio, then normalize
    bce_weight_ratio = trial.suggest_float("bce_weight_ratio", 0.1, 1.0)
    dice_weight_ratio = trial.suggest_float("dice_weight_ratio", 0.1, 1.0)
    total_weight = bce_weight_ratio + dice_weight_ratio
    bce_weight = bce_weight_ratio / total_weight
    dice_weight = dice_weight_ratio / total_weight

    # Optional: Tune base_filters or batch_size if desired (more computationally expensive)
    # base_filters = trial.suggest_categorical("base_filters", [32, 64, 96])
    # batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    base_filters = args.base_filters # Keep fixed for now
    batch_size = args.batch_size     # Keep fixed for now

    print(f"\n--- Trial {trial.number} ---")
    print(f"  Params: LR={lr:.1e}, WD={weight_decay:.1e}, BCE_W={bce_weight:.3f}, DICE_W={dice_weight:.3f}")

    # --- Load Data (same split each trial due to fixed seed) ---
    if args.model_type == 'csi': DatasetClass = CsiDataset; input_channels = 156; data_dir = args.csi_data_dir
    elif args.model_type == 'rssi': DatasetClass = RssiDataset; input_channels = 3; data_dir = args.rssi_data_dir
    else: raise ValueError("Invalid model_type")
    if not Path(data_dir).is_dir(): raise FileNotFoundError(f"Data dir not found: {data_dir}")

    full_dataset = DatasetClass(data_dir=data_dir)
    if len(full_dataset) == 0: raise ValueError("No samples found")
    total_size = len(full_dataset); test_size = int(total_size * args.test_split); val_size = int(total_size * args.val_split); train_size = total_size - val_size - test_size
    if train_size <= 0 or val_size <= 0: raise ValueError("Dataset split invalid size")
    train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Setup Model, Loss, Optimizer ---
    model = SignalToMaskUNet(input_channels=input_channels, base_filters=base_filters).to(device)
    criterion_bce = nn.BCELoss()
    criterion_dice = DiceLoss(epsilon=args.dice_epsilon)
    if args.optimizer.lower() == 'adam': optimizer = optim.Adam(model.parameters(), lr=lr)
    elif args.optimizer.lower() == 'adamw': optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else: raise ValueError("Invalid optimizer")

    # --- Training Loop for Tuning ---
    best_val_loss = float('inf')
    epochs_to_tune = args.tune_epochs # Use fewer epochs for tuning

    for epoch in range(epochs_to_tune):
        train_loss = train_one_epoch_tuned(model, train_loader, criterion_bce, criterion_dice, bce_weight, dice_weight, optimizer, device)
        val_loss = validate_tuned(model, val_loader, criterion_bce, criterion_dice, bce_weight, dice_weight, device)

        print(f"  Epoch [{epoch+1}/{epochs_to_tune}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Update best validation loss for this trial
        best_val_loss = min(best_val_loss, val_loss)

        # --- Optuna Pruning ---
        trial.report(val_loss, epoch) # Report intermediate value
        if trial.should_prune():
            print(f"  Trial {trial.number} pruned at epoch {epoch+1}.")
            raise optuna.exceptions.TrialPruned()
        # ----------------------

    print(f"  Trial {trial.number} finished. Best Val Loss: {best_val_loss:.6f}")
    return best_val_loss # Optuna minimizes this value

# --- Script Entry Point ---
if __name__ == "__main__":
    if not add_project_root_to_path(): sys.exit(1)
    from neural_netwok.csi_dataset import CsiDataset; from neural_netwok.rssi_dataset import RssiDataset
    from neural_netwok.model import SignalToMaskUNet, DiceLoss

    script_dir = Path(__file__).parent.resolve(); project_root = script_dir.parent
    default_processed_csi = project_root / "processed_data_csi"; default_processed_rssi = project_root / "processed_data_rssi"
    default_output_dir = project_root / "tuning_output" # Separate output for tuning results

    parser = argparse.ArgumentParser(description="Tune hyperparameters for SignalToMaskUNet.")
    # Args needed for objective function setup
    parser.add_argument('--model_type', type=str, required=True, choices=['csi', 'rssi'], help="Type of data/model.")
    parser.add_argument('--csi_data_dir', type=str, default=str(default_processed_csi), help="Path to processed CSI data.")
    parser.add_argument('--rssi_data_dir', type=str, default=str(default_processed_rssi), help="Path to processed RSSI data.")
    parser.add_argument('--base_filters', type=int, default=64, help="Base filters (fixed during tuning unless added to objective).")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size (fixed during tuning unless added to objective).")
    parser.add_argument('--val_split', type=float, default=0.15, help="Validation split fraction.")
    parser.add_argument('--test_split', type=float, default=0.15, help="Test split fraction (defines dataset size).")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader workers.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for dataset split.")
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help="Optimizer type.")
    parser.add_argument('--dice_epsilon', type=float, default=1e-6, help="Epsilon for Dice loss.")
    # parser.add_argument('--smoothing_window', type=int, default=1, help="Online smoothing window.")
    # Optuna Args
    parser.add_argument('--n_trials', type=int, default=50, help="Number of Optuna trials to run.")
    parser.add_argument('--tune_epochs', type=int, default=15, help="Number of epochs to train each trial.")
    parser.add_argument('--study_name', type=str, default=None, help="Optuna study name (optional).")
    parser.add_argument('--storage', type=str, default=None, help="Optuna storage URL (e.g., sqlite:///tuning.db) (optional).")
    parser.add_argument('--output_dir', type=str, default=str(default_output_dir), help="Directory to save tuning results.")


    args = parser.parse_args()
    if args.val_split + args.test_split >= 1.0: print("ERROR: Val + Test split must be < 1.0"); sys.exit(1)

    # Create output directory for tuning study
    tune_output_dir = Path(args.output_dir) / args.model_type
    tune_output_dir.mkdir(parents=True, exist_ok=True)
    if args.storage is None:
         # Default to sqlite file in output dir if no storage specified
         args.storage = f"sqlite:///{tune_output_dir}/optuna_study_{args.model_type}.db"
         print(f"Using default Optuna storage: {args.storage}")

    # --- Run Optuna Study ---
    study = optuna.create_study(
        study_name=args.study_name if args.study_name else f"{args.model_type}_tuning_{datetime.now().strftime('%Y%m%d')}",
        direction="minimize", # Minimize validation loss
        storage=args.storage,
        load_if_exists=True, # Resume study if it exists
        pruner=optuna.pruners.MedianPruner() # Example pruner
    )

    # Pass args to the objective function using a lambda
    objective_with_args = lambda trial: objective(trial, args)

    print(f"Starting Optuna study: {study.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Epochs per trial: {args.tune_epochs}")

    try:
        study.optimize(objective_with_args, n_trials=args.n_trials, timeout=args.timeout if hasattr(args, 'timeout') else None) # Add timeout if needed
    except KeyboardInterrupt:
         print("Tuning stopped manually.")

    # --- Print Results ---
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\n--- Optuna Study Statistics ---")
    print(f"  Study name: {study.study_name}")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    if complete_trials: # Check if any trials completed successfully
        best_trial = study.best_trial
        print("\n--- Best Trial ---")
        print(f"  Value (Best Val Loss): {best_trial.value:.6f}")
        print("  Params: ")
        # Recalculate final weights from best trial params for saving
        best_params = best_trial.params
        final_bce_weight = 0.5 # Default if ratios not tuned
        final_dice_weight = 0.5
        if "bce_weight_ratio" in best_params and "dice_weight_ratio" in best_params:
             bce_ratio = best_params["bce_weight_ratio"]
             dice_ratio = best_params["dice_weight_ratio"]
             total_w = bce_ratio + dice_ratio
             if total_w > 1e-6: # Avoid division by zero
                 final_bce_weight = bce_ratio / total_w
                 final_dice_weight = dice_ratio / total_w

        # Print and save params including calculated weights
        params_to_save = {}
        for key, value in best_params.items():
            # Exclude the ratios from the final saved params if weights are calculated
            if "weight_ratio" not in key:
                params_to_save[key] = value
                if isinstance(value, float): print(f"    {key}: {value:.6f}")
                else: print(f"    {key}: {value}")
        # Add the calculated final weights
        params_to_save["bce_weight"] = final_bce_weight
        params_to_save["dice_weight"] = final_dice_weight
        print(f"    bce_weight: {final_bce_weight:.6f} (calculated)")
        print(f"    dice_weight: {final_dice_weight:.6f} (calculated)")


        # Save best params to a file
        best_params_path = tune_output_dir / f"best_params_{args.model_type}.txt"
        with open(best_params_path, 'w') as f:
             f.write(f"# Best hyperparameters for {args.model_type} from study '{study.study_name}'\n")
             f.write(f"# Best Validation Loss: {best_trial.value:.8f}\n")
             # Save the cleaned/calculated params
             for key, value in params_to_save.items():
                  f.write(f"--{key.replace('_', '-')} {value}\n") # Format for command line use
             # Also add fixed params for completeness
             f.write(f"\n# Fixed parameters during tuning:\n")
             f.write(f"--base-filters {args.base_filters}\n")
             f.write(f"--batch-size {args.batch_size}\n")
             f.write(f"--optimizer {args.optimizer}\n")
             # ... add others if needed ...
        print(f"\nBest parameters saved to: {best_params_path}")

    else:
        print("\nNo trials completed successfully.")

    print("-----------------------------")