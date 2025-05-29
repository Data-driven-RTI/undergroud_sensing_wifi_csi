# scripts/train_model.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from pathlib import Path
import argparse
import time
import random
import numpy as np
import traceback
from datetime import datetime # Added for timestamp

# --- Helper Function to Add Project Root to Path ---
# (Keep the add_project_root_to_path function as before)
def add_project_root_to_path():
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        print(f"Adding project root to Python path: {project_root}")
        sys.path.insert(0, str(project_root))
    try:
        from neural_netwok.csi_dataset import CsiDataset
        from neural_netwok.rssi_dataset import RssiDataset
        # from neural_netwok.model import SignalToMaskCNN
        from neural_netwok.model import SignalToMaskUNet, CombinedLoss
        print("Successfully imported Dataset and Model classes.")
        return True
    except ModuleNotFoundError as e:
        print("\n--- ERROR ---")
        print(f"Could not import required classes: {e}")
        print("Please ensure:")
        print("  1. This script ('train_model.py') is in the 'scripts/' directory.")
        print("  2. The 'neural_netwok/' directory exists in the project root.")
        print("  3. 'csi_dataset.py', 'rssi_dataset.py', and 'model.py' are inside 'neural_netwok/'.")
        print(f"Current sys.path: {sys.path}")
        print("-------------")
        return False
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An unexpected error occurred during import: {e}")
        traceback.print_exc()
        print("-------------")
        return False

# --- Set Random Seed ---
# (Keep the set_seed function as before)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Training & Validation Functions ---
# (Keep train_one_epoch and validate functions as before)
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 50 == 0 or (i + 1) == len(dataloader):
             batch_time = time.time() - start_time
             print(f'Epoch [{epoch+1}/{total_epochs}], Batch [{i+1}/{len(dataloader)}], '
                   f'Loss: {loss.item():.4f}, Time: {batch_time:.2f}s')
             start_time = time.time()
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    val_loss = running_loss / len(dataloader)
    return val_loss

# --- Function to Save Split Information ---
def save_split_info(output_dir, train_subset, val_subset, test_subset):
    """Saves the file paths for train, validation, and test sets."""
    split_file_path = output_dir / "split_info.txt"
    print(f"Saving data split information to: {split_file_path}")

    def get_paths_from_subset(subset):
        # Access the original dataset and indices from the Subset object
        original_dataset = subset.dataset
        indices = subset.indices
        # Retrieve the list of (feature_path, gt_path) tuples for these indices
        return [original_dataset.samples[i] for i in indices]

    train_paths = get_paths_from_subset(train_subset)
    val_paths = get_paths_from_subset(val_subset)
    test_paths = get_paths_from_subset(test_subset)

    try:
        with open(split_file_path, 'w') as f:
            f.write("# Training Set Files\n")
            for feat_path, gt_path in train_paths:
                f.write(f"TRAIN\t{feat_path}\t{gt_path}\n")

            f.write("\n# Validation Set Files\n")
            for feat_path, gt_path in val_paths:
                f.write(f"VAL\t{feat_path}\t{gt_path}\n")

            f.write("\n# Test Set Files\n")
            for feat_path, gt_path in test_paths:
                f.write(f"TEST\t{feat_path}\t{gt_path}\n")
        print("Split information saved successfully.")
    except Exception as e:
        print(f"Error saving split information: {e}")
        traceback.print_exc()


# --- Main Execution ---
def main(args):
    # --- Setup ---
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / args.model_type / timestamp # Changed path
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / f"training_log_{args.model_type}_{timestamp}.txt" # Changed log name

    # Basic logging setup
    def log_message(message):
        print(message)
        with open(log_file_path, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n") # Added timestamp to log entries

    log_message("--- Training Configuration ---")
    for arg, value in vars(args).items():
        log_message(f"{arg}: {value}")
    log_message(f"Timestamp: {timestamp}") # Log timestamp
    log_message(f"Output directory: {output_dir}")
    log_message(f"Log file: {log_file_path}")
    log_message("----------------------------")

    # --- Load Data ---
    if args.model_type == 'csi':
        DatasetClass = CsiDataset
        input_channels = 156
        data_dir = args.csi_data_dir
    elif args.model_type == 'rssi':
        DatasetClass = RssiDataset
        input_channels = 3
        data_dir = args.rssi_data_dir
    else:
        log_message("ERROR: Invalid model_type. Choose 'csi' or 'rssi'.")
        sys.exit(1)

    if not Path(data_dir).is_dir():
         log_message(f"ERROR: Data directory not found: {data_dir}")
         sys.exit(1)

    log_message(f"Loading {args.model_type.upper()} data from: {data_dir}")
    
    # Load the full dataset first to get all sample paths
    full_dataset = DatasetClass(data_dir=data_dir)

    if len(full_dataset) == 0:
        log_message("ERROR: No samples found in the dataset. Check data directory and dataset implementation.")
        sys.exit(1)

    # Split dataset into Train, Validation, Test
    total_size = len(full_dataset)
    test_size = int(total_size * args.test_split)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size - test_size

    if train_size <= 0 or val_size <= 0 or test_size <= 0:
         log_message(f"ERROR: Dataset split resulted in non-positive set size (Total: {total_size}, Train: {train_size}, Val: {val_size}, Test: {test_size}). Adjust splits or get more data.")
         sys.exit(1)

    log_message(f"Total samples: {total_size}, Training: {train_size}, Validation: {val_size}, Test: {test_size}")

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed) # Use generator for reproducibility
    )

    # Save the split information (file paths)
    save_split_info(output_dir, train_dataset, val_dataset, test_dataset)

    # Create DataLoaders (only for train and val during training)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True if device == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True if device == 'cuda' else False)
    # Note: test_loader is not created here, it will be created in the test script
    log_message("Train/Val DataLoaders created.")

    # --- Initialize Model, Loss, Optimizer ---
    log_message(f"Initializing model with input_channels={input_channels}, base_filters={args.base_filters}")
    model = SignalToMaskUNet(input_channels=input_channels, base_filters=args.base_filters).to(device)

    log_message("Setting up CombinedLoss (BCE + Dice)")
    # Define weights for the combined loss (you can make these argparse arguments too)
    bce_weight = 0.5  # Example: Equal weighting
    dice_weight = 0.5 # Example: Equal weighting
    dice_smooth = 1e-6 # Smoothing factor for Dice Loss stability

    # Instantiate the CombinedLoss
    # criterion = CombinedLoss(bce_weight=bce_weight,
    #                          dice_weight=dice_weight,
    #                          smooth=dice_smooth).to(device) # Move criterion to the correct device!

    criterion = nn.BCELoss().to(device)
    # Keep optimizer as AdamW or switch back to Adam if preferred
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, verbose=True)

    log_message(f"Model:\n{model}") # Consider removing for very large models
    log_message(f"Criterion: {criterion}")
    log_message(f"Optimizer: {optimizer}")

    # --- Training Loop ---
    best_val_loss = float('inf')
    best_model_path = output_dir / f"best_model_{args.model_type}.pth" # Model saved inside timestamped folder

    log_message("\n--- Starting Training ---")
    start_train_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        current_lr = optimizer.param_groups[0]['lr']

        log_message(f"Epoch [{epoch+1}/{args.epochs}] - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Duration: {epoch_duration:.2f}s, lr: {current_lr}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            log_message(f"  -> New best model saved to {best_model_path} (Val Loss: {best_val_loss:.4f})")

    end_train_time = time.time()
    total_train_time = end_train_time - start_train_time
    log_message("--- Training Finished ---")
    log_message(f"Total training time: {total_train_time:.2f} seconds")
    log_message(f"Best validation loss: {best_val_loss:.4f}")
    log_message(f"Best model saved at: {best_model_path}")
    log_message(f"Split info saved at: {output_dir / 'split_info.txt'}")
    log_message("-------------------------")


if __name__ == "__main__":
    if not add_project_root_to_path():
        sys.exit(1)

    from neural_netwok.csi_dataset import CsiDataset
    from neural_netwok.rssi_dataset import RssiDataset
    # from neural_netwok.model import SignalToMaskCNN
    from neural_netwok.model import SignalToMaskUNet, CombinedLoss


    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    default_processed_csi = project_root / "processed_data_csi"
    default_processed_rssi = project_root / "processed_data_rssi"
    default_output_dir = project_root / "training_output"

    parser = argparse.ArgumentParser(description="Train SignalToMaskUNet model on CSI or RSSI data.")

    parser.add_argument('--model_type', type=str, required=True, choices=['csi', 'rssi'],
                        help="Type of data/model to train ('csi' or 'rssi').")
    parser.add_argument('--csi_data_dir', type=str, default=str(default_processed_csi),
                        help=f"Path to processed CSI data directory (default: {default_processed_csi})")
    parser.add_argument('--rssi_data_dir', type=str, default=str(default_processed_rssi),
                        help=f"Path to processed RSSI data directory (default: {default_processed_rssi})")
    parser.add_argument('--output_dir', type=str, default=str(default_output_dir),
                        help=f"Base directory to save models and logs (default: {default_output_dir}). Timestamped subfolder will be created.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size (default: 16)")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument('--base_filters', type=int, default=64, help="Base number of filters in the CNN model (default: 64)")
    parser.add_argument('--val_split', type=float, default=0.15, help="Fraction of data for validation (default: 0.15)")
    parser.add_argument('--test_split', type=float, default=0.15, help="Fraction of data for testing (default: 0.15)") # Added test split arg
    parser.add_argument('--num_workers', type=int, default=4, help="Number of worker processes for DataLoader (default: 4)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument('--bce_weight', type=float, default=0.5, help="Weight for BCE component in CombinedLoss (default: 0.5)")
    parser.add_argument('--dice_weight', type=float, default=0.5, help="Weight for Dice component in CombinedLoss (default: 0.5)")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay for AdamW/Adam optimizer (default: 0.0)")
    args = parser.parse_args()

    if not np.isclose(args.bce_weight + args.dice_weight, 1.0) and (args.bce_weight > 0 or args.dice_weight > 0):
         print(f"Warning: BCE weight ({args.bce_weight}) + Dice weight ({args.dice_weight}) does not sum to 1. Adjust if needed.")
         
    # Validate splits sum to <= 1.0
    if args.val_split + args.test_split >= 1.0:
        print(f"ERROR: Validation split ({args.val_split}) + Test split ({args.test_split}) must be less than 1.0")
        sys.exit(1)

    main(args)