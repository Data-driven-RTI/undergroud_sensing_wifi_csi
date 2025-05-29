import os
import re
import csv
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import traceback

# --- Configuration ---
TARGET_CHANNELS = [1, 6, 11] # Channels to extract RSSI from
MAX_NODES = 16             # Number of nodes (0 to 15)
NUM_TARGET_CHANNELS = len(TARGET_CHANNELS)
# --- Imputation ---
# Choose how to handle missing links/channels in the final matrix
# Options: 'zero', 'constant'
IMPUTATION_METHOD = 'constant'
MISSING_VALUE_FILL = -100.0 # Value to use if IMPUTATION_METHOD is 'constant'
# --------------------

# --- Main Processing Function for One Recording ---
def process_recording_session_rssi(csi_csv_path, gt_npy_path, output_dir):
    """Processes a single recording session for RSSI features."""
    print(f"  Processing RSSI for: {csi_csv_path.name}")
    timestamp = csi_csv_path.stem.replace("csi_data_", "")

    # --- Phase 1: Load CSV and Extract RSSI ---
    try:
        # Specify columns to load for efficiency
        columns_to_load = ['sender_node', 'receiver_node', 'channel_index', 'rssi']
        df = pd.read_csv(csi_csv_path, usecols=columns_to_load, low_memory=False)
        if df.empty:
            print(f"    Warning: CSV file is empty or missing required columns. Skipping.")
            return False
        # Convert columns to appropriate types, handle potential errors
        df = df.astype({'sender_node': int, 'receiver_node': int, 'channel_index': int, 'rssi': int})

    except FileNotFoundError:
        print(f"    Error: CSV file not found. Skipping.")
        return False
    except ValueError as e:
         print(f"    Error: Could not convert columns to integers in {csi_csv_path.name}. Check CSV format. Error: {e}")
         return False
    except KeyError as e:
         print(f"    Error: Missing required column {e} in {csi_csv_path.name}. Skipping.")
         return False
    except Exception as e:
        print(f"    Error reading CSV file {csi_csv_path}: {e}")
        traceback.print_exc()
        return False

    # Filter for target channels only
    df_filtered = df[df['channel_index'].isin(TARGET_CHANNELS)].copy()

    if df_filtered.empty:
        print(f"    Warning: No data found for target channels {TARGET_CHANNELS}. Skipping file.")
        return False

    # --- Phase 2: Aggregate (Average) RSSI ---
    # Group by sender, receiver, channel and calculate mean RSSI
    try:
        # Exclude self-links before grouping
        df_filtered = df_filtered[df_filtered['sender_node'] != df_filtered['receiver_node']]
        if df_filtered.empty:
             print(f"    Warning: No non-self-link data found for target channels. Skipping file.")
             return False

        averaged_rssi_series = df_filtered.groupby(['sender_node', 'receiver_node', 'channel_index'])['rssi'].mean()
        averaged_rssi = averaged_rssi_series.to_dict() # Convert to dictionary {(s, r, ch): avg_rssi}
    except Exception as e:
        print(f"    Error during RSSI aggregation: {e}")
        traceback.print_exc()
        return False

    if not averaged_rssi:
         print(f"    Warning: No valid averaged RSSI data after grouping. Skipping file.")
         return False

    # --- Phase 3 & 4: Construct Feature Matrix & Impute ---
    # Initialize matrix based on imputation choice
    if IMPUTATION_METHOD == 'zero':
        feature_matrix = np.zeros((NUM_TARGET_CHANNELS, MAX_NODES, MAX_NODES), dtype=np.float32)
        print(f"    Using Zero Filling for missing data.")
    elif IMPUTATION_METHOD == 'constant':
        feature_matrix = np.full((NUM_TARGET_CHANNELS, MAX_NODES, MAX_NODES), MISSING_VALUE_FILL, dtype=np.float32)
        print(f"    Using Constant Value ({MISSING_VALUE_FILL}) Filling for missing data.")
    else: # Default to zero if method is unknown
        feature_matrix = np.zeros((NUM_TARGET_CHANNELS, MAX_NODES, MAX_NODES), dtype=np.float32)
        print(f"    Warning: Unknown IMPUTATION_METHOD '{IMPUTATION_METHOD}'. Defaulting to Zero Filling.")


    channel_to_feature_idx = {ch: i for i, ch in enumerate(TARGET_CHANNELS)} # {1: 0, 6: 1, 11: 2}

    for (sender, receiver, channel), avg_rssi in averaged_rssi.items():
        if channel in channel_to_feature_idx: # Check if it's a target channel
            feature_idx = channel_to_feature_idx[channel]
            # Ensure indices are within bounds
            if 0 <= sender < MAX_NODES and 0 <= receiver < MAX_NODES:
                feature_matrix[feature_idx, sender, receiver] = float(avg_rssi) # Assign average RSSI
            else:
                 print(f"    Warning: Sender ({sender}) or Receiver ({receiver}) out of bounds during matrix fill.")

    # Set diagonal (self-links) explicitly to the fill value if using constant fill
    if IMPUTATION_METHOD == 'constant':
        for i in range(NUM_TARGET_CHANNELS):
            np.fill_diagonal(feature_matrix[i, :, :], MISSING_VALUE_FILL)


    # --- Phase 4: Saving ---
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths
        output_rssi_path = output_dir / "rssi_features.npy" # Changed filename
        output_gt_path = output_dir / "ground_truth.npy"

        # Save features
        np.save(output_rssi_path, feature_matrix)
        print(f"      Saved RSSI features: {output_rssi_path}")

        # Copy ground truth
        shutil.copy2(gt_npy_path, output_gt_path)
        print(f"      Copied ground truth: {output_gt_path}")

        return True # Indicate success

    except Exception as e:
        print(f"    Error saving processed files for {timestamp}: {e}")
        traceback.print_exc()
        return False


# --- Main Orchestration Function ---
def process_dataset_rssi(raw_data_dir, processed_data_dir_rssi): # Note changed output dir name
    """Finds matching files and processes each recording session for RSSI."""
    print(f"Starting RSSI data processing...")
    print(f"Raw data source: {raw_data_dir}")
    print(f"Processed data destination: {processed_data_dir_rssi}") # Use specific RSSI dir

    raw_data_path = Path(raw_data_dir).resolve()
    processed_data_path = Path(processed_data_dir_rssi).resolve() # Use specific RSSI dir

    if not raw_data_path.is_dir():
        print(f"Error: Raw data directory not found: {raw_data_path}")
        return

    # Find potato_X directories
    potato_dirs = sorted([d for d in raw_data_path.iterdir() if d.is_dir() and re.match(r'^potato_\d+$', d.name)])
    if not potato_dirs:
        print(f"Warning: No 'potato_X' directories found in {raw_data_path}")
        return

    total_processed = 0
    total_skipped_missing = 0
    total_errors_processing = 0

    for potato_dir in potato_dirs:
        dataset_name = potato_dir.name
        gt_dir_name = f"{dataset_name}_gt"
        gt_dir_path = raw_data_path / gt_dir_name

        print(f"\nProcessing dataset: {dataset_name}")

        if not gt_dir_path.is_dir():
            print(f"  Warning: Ground truth directory not found, skipping dataset: {gt_dir_path}")
            continue

        # Find rec_YYYYMMDD_HHMMSS directories
        recording_dirs = sorted([d for d in potato_dir.iterdir() if d.is_dir() and d.name.startswith('rec_')])
        if not recording_dirs:
            print(f"  Warning: No 'rec_...' subdirectories found in {potato_dir}")
            continue

        print(f"  Found {len(recording_dirs)} recording directories.")

        for rec_dir in recording_dirs:
            match = re.match(r'^rec_(\d{8}_\d{6})$', rec_dir.name)
            if not match: continue

            timestamp = match.group(1)
            csi_file_name = f"csi_data_{timestamp}.csv"
            gt_file_name = f"{timestamp}.npy"
            src_csi_path = rec_dir / csi_file_name
            src_gt_path = gt_dir_path / gt_file_name

            # Check existence before processing
            if not src_csi_path.is_file():
                print(f"    Skipping {timestamp}: CSI file missing ({src_csi_path.name})")
                total_skipped_missing += 1
                continue
            if not src_gt_path.is_file():
                print(f"    Skipping {timestamp}: Ground truth file missing ({src_gt_path.name})")
                total_skipped_missing += 1
                continue

            # Define output directory for this recording in the RSSI processed folder
            output_rec_dir = processed_data_path / dataset_name / timestamp

            # Process this recording session using the RSSI function
            success = process_recording_session_rssi(src_csi_path, src_gt_path, output_rec_dir)

            if success:
                total_processed += 1
            else:
                total_errors_processing += 1

    print("\n------------------------------------")
    print("RSSI Data processing complete.")
    print(f"Total recordings processed successfully: {total_processed}")
    print(f"Total recordings skipped (missing files): {total_skipped_missing}")
    print(f"Total recordings with errors during processing: {total_errors_processing}")
    print("------------------------------------")


# --- Script Entry Point ---
if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    default_raw_data = project_root / "raw_data"
    # *** Define default output directory for RSSI data ***
    default_processed_data_rssi = project_root / "processed_data_rssi"

    parser = argparse.ArgumentParser(description="Process raw CSI CSV data into RSSI feature matrices.")
    parser.add_argument("--raw_dir", type=str, default=str(default_raw_data), help=f"Path to raw_data directory (default: {default_raw_data})")
    # *** Update argument name and default path ***
    parser.add_argument("--processed_dir_rssi", type=str, default=str(default_processed_data_rssi), help=f"Path to processed_data_rssi directory (default: {default_processed_data_rssi})")
    args = parser.parse_args()

    # Create processed data directory if it doesn't exist
    Path(args.processed_dir_rssi).mkdir(parents=True, exist_ok=True)

    # Call the RSSI processing function
    process_dataset_rssi(args.raw_dir, args.processed_dir_rssi)