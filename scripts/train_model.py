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
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau


def add_project_root_to_path():
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from neural_netwok.csi_dataset import CsiDataset
        from neural_netwok.rssi_dataset import RssiDataset
        from neural_netwok.model import SignalToMaskUNet, AdaptedFCN, AdaptedDeepLabV3
        return True
    except ModuleNotFoundError as e:
        sys.exit(1)
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An unexpected error occurred during import: {e}")
        traceback.print_exc()
        print("-------------")
        return False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def save_split_info(output_dir, train_subset, val_subset, test_subset):
    split_file_path = output_dir / "split_info.txt"
    print(f"Saving data split information to: {split_file_path}")

    def get_paths_from_subset(subset):
        original_dataset = subset.dataset
        indices = subset.indices
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


def run_training_instance(
    model_type: str,
    architecture: str,
    gpu_id: int,
    base_output_dir: Path,
    csi_data_dir: Path,
    rssi_data_dir: Path,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    base_filters: int = 64,
    val_split: float = 0.15,
    test_split: float = 0.15,
    weight_decay: float = 0.0,
    scheduler_patience: int = 7,
    scheduler_factor: float = 0.5,
    num_workers: int = 4
    ):
    run_start_time = time.time()
    print(f"--- Starting Run: {model_type.upper()}-{architecture.upper()} on GPU {gpu_id} ---")

    set_seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:0")
        print(f"Process for {model_type}-{architecture} assigned to use Physical GPU {gpu_id} (seen as {device})")
    else:
        device = torch.device("cpu")
        print(f"Process for {model_type}-{architecture} using CPU (CUDA not available or GPU assignment failed)")


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / model_type / architecture / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / f"training_log_{model_type}_{architecture}_{timestamp}.txt"


    def log_message(message):
        prefix = f"[{model_type.upper()}-{architecture.upper()}-GPU{gpu_id}]"
        print(f"{prefix} {message}")
        with open(log_file_path, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    log_message("--- Training Configuration ---")
    config = locals()
    for key, value in config.items():
         if key not in ['output_dir', 'log_file_path', 'log_message', 'device', 'run_start_time', 'prefix']:
              log_message(f"{key}: {value}")
    log_message(f"Output directory: {output_dir}")
    log_message("----------------------------")


    try:
        add_project_root_to_path()
        from neural_netwok.csi_dataset import CsiDataset
        from neural_netwok.rssi_dataset import RssiDataset
        from neural_netwok.model import SignalToMaskUNet, AdaptedFCN, AdaptedDeepLabV3
    except ImportError:
        log_message("ERROR: Failed to import modules inside run_training_instance. Check paths.")
        return

    if model_type == 'csi':
        DatasetClass = CsiDataset
        input_channels = 156
        data_dir = csi_data_dir
    elif model_type == 'rssi':
        DatasetClass = RssiDataset
        input_channels = 3
        data_dir = rssi_data_dir
    else:
        log_message("ERROR: Invalid model_type.")
        return

    if not Path(data_dir).is_dir():
         log_message(f"ERROR: Data directory not found: {data_dir}")
         return

    log_message(f"Loading {model_type.upper()} data from: {data_dir}")
    try:
        full_dataset = DatasetClass(data_dir=data_dir)
    except Exception as e:
        log_message(f"ERROR: Failed to load dataset {DatasetClass} from {data_dir}: {e}")
        traceback.print_exc(file=open(log_file_path, 'a'))
        return

    if len(full_dataset) == 0:
        log_message("ERROR: No samples found in the dataset.")
        return

    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size

    if train_size <= 0 or val_size <= 0 or test_size <= 0:
         log_message(f"ERROR: Dataset split resulted in non-positive set size.")
         return

    log_message(f"Splitting: Total={total_size}, Train={train_size}, Val={val_size}, Test={test_size}")
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    try:
        save_split_info(output_dir, train_dataset, val_dataset, test_dataset)
    except Exception as e:
        log_message(f"ERROR saving split info: {e}")

    effective_num_workers = 0
    log_message(f"Creating DataLoaders with num_workers={effective_num_workers} (forced for multiprocessing Pool)")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=effective_num_workers,
                              pin_memory=True if device.type == 'cuda' else False,
                              persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=effective_num_workers,
                            pin_memory=True if device.type == 'cuda' else False,
                            persistent_workers=False)
    log_message("Train/Val DataLoaders created.")
    log_message("Train/Val DataLoaders created.")

    log_message(f"Initializing {architecture} model with input_channels={input_channels}, base_filters={base_filters}")
    try:
        if architecture == 'unet':
            model = SignalToMaskUNet(input_channels=input_channels, base_filters=base_filters).to(device)
        elif architecture == 'fcn':
            model = AdaptedFCN(input_channels=input_channels, base_filters=base_filters).to(device)
        elif architecture == 'deeplab':
            model = AdaptedDeepLabV3(input_channels=input_channels, base_filters=base_filters).to(device)
        else:
            log_message(f"ERROR: Unknown architecture '{architecture}'")
            return
    except Exception as e:
        log_message(f"ERROR Initializing model {architecture}: {e}")
        traceback.print_exc(file=open(log_file_path, 'a'))
        return


    log_message("Using BCELoss")
    criterion = nn.BCELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=False)

    log_message(f"Criterion: {criterion}")
    log_message(f"Optimizer: {optimizer}")
    log_message(f"Scheduler: factor={scheduler_factor}, patience={scheduler_patience}")

    best_val_loss = float('inf')
    best_model_path = output_dir / f"best_model_{model_type}_{architecture}.pth"

    log_message("\n--- Starting Training Loop ---")
    loop_start_time = time.time()

    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
            val_loss = validate(model, val_loader, criterion, device)

            if np.isnan(train_loss) or np.isnan(val_loss):
                log_message(f"ERROR: NaN loss detected at epoch {epoch+1}. Stopping training for this run.")
                break

            scheduler.step(val_loss)
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']

            log_message(f"Epoch [{epoch+1}/{epochs}] - "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                        f"Duration: {epoch_duration:.2f}s, lr: {current_lr:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                log_message(f"  -> New best model saved (Val Loss: {best_val_loss:.4f})")

    except Exception as e:
         log_message(f"ERROR during training loop: {e}")
         traceback.print_exc(file=open(log_file_path, 'a'))
    finally:
        loop_end_time = time.time()
        total_train_time = loop_end_time - loop_start_time
        run_duration = loop_end_time - run_start_time
        log_message("--- Training Loop Finished ---")
        log_message(f"Training loop duration: {total_train_time:.2f} seconds")
        log_message(f"Best validation loss: {best_val_loss:.4f}")
        if best_model_path.exists():
            log_message(f"Best model saved at: {best_model_path}")
        else:
            log_message("Best model was not saved (no improvement or error).")
        log_message(f"Total Run Duration: {run_duration:.2f} seconds")
        log_message(f"--- Finished Run: {model_type.upper()}-{architecture.upper()} on GPU {gpu_id} ---")


if __name__ == "__main__":
    print("This script is primarily designed to be called by run_all_experiments.py")
    print("Attempting to run standalone using argparse...")

    if not add_project_root_to_path():
        sys.exit(1)

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    default_processed_csi = project_root / "processed_data_csi"
    default_processed_rssi = project_root / "processed_data_rssi"
    default_output_dir = project_root / "training_output_standalone"

    parser = argparse.ArgumentParser(description="Train a single model instance (primarily for debugging).")

    parser.add_argument('--model_type', type=str, required=True, choices=['csi', 'rssi'])
    parser.add_argument('--architecture', type=str, required=True, choices=['unet', 'fcn', 'deeplab'])
    parser.add_argument('--gpu_id', type=int, required=True, help="GPU ID to use.")
    parser.add_argument('--output_dir', type=str, default=str(default_output_dir), help="Base directory for output.")
    parser.add_argument('--csi_data_dir', type=str, default=str(default_processed_csi))
    parser.add_argument('--rssi_data_dir', type=str, default=str(default_processed_rssi))
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base_filters', type=int, default=64)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--test_split', type=float, default=0.15)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--scheduler_patience', type=int, default=7)
    parser.add_argument('--scheduler_factor', type=float, default=0.5)

    args = parser.parse_args()

    run_training_instance(
        model_type=args.model_type,
        architecture=args.architecture,
        gpu_id=args.gpu_id,
        base_output_dir=Path(args.output_dir),
        csi_data_dir=Path(args.csi_data_dir),
        rssi_data_dir=Path(args.rssi_data_dir),
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        base_filters=args.base_filters,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor
    )
    if not add_project_root_to_path():
        sys.exit(1)

    from neural_netwok.csi_dataset import CsiDataset
    from neural_netwok.rssi_dataset import RssiDataset
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
         
    if args.val_split + args.test_split >= 1.0:
        print(f"ERROR: Validation split ({args.val_split}) + Test split ({args.test_split}) must be less than 1.0")
        sys.exit(1)

    main(args)