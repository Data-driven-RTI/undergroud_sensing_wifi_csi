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
import time # For basic timing info

# --- Configuration ---

# !!=========================================================================!!
# !! Indices for the 52 commonly used DATA subcarriers in HT20               !!
# !! Assumes the 64 complex values correspond to subcarriers -32 to +31.     !!
# !! Excludes DC (0) and standard pilots (-21, -7, +7, +21).                 !!
# !!=========================================================================!!
_HT20_PILOT_OFFSETS = {-21, -7, 7, 21}
_HT20_NULL_DC_OFFSET = {0}
_all_potential_data_pilot_offsets = set(range(-28, 29)) # -28 to +28 inclusive
_data_relative_offsets = _all_potential_data_pilot_offsets - _HT20_NULL_DC_OFFSET - _HT20_PILOT_OFFSETS
_center_index_in_64 = 32 # Array index corresponding to subcarrier 0 (DC)
DATA_SUBCARRIER_INDICES = sorted(
    [_center_index_in_64 + offset for offset in _data_relative_offsets]
)
# --- >>>>>>>> VERIFICATION SHOULD PASS NOW <<<<<<<< ---

# --- Sanity Check ---
# *** CORRECTED THIS CHECK TO EXPECT 52 ***
if len(DATA_SUBCARRIER_INDICES) != 52:
    print(f"ERROR: DATA_SUBCARRIER_INDICES length is {len(DATA_SUBCARRIER_INDICES)}, expected 52. Check definition.")
    print(f"Generated Indices ({len(DATA_SUBCARRIER_INDICES)}): {DATA_SUBCARRIER_INDICES}")
    exit(1)
else:
    print(f"Using {len(DATA_SUBCARRIER_INDICES)} subcarrier indices for CSI processing.")
# --------------------

TARGET_CHANNELS = [1, 6, 11] # Channels for both CSI and RSSI features
MAX_NODES = 16             # Number of nodes (0 to 15)

# CSI Config
NUM_CSI_SUBCARRIERS = 52       # *** CORRECTED TO 52 ***
NUM_CSI_TARGET_CHANNELS = len(TARGET_CHANNELS)
NUM_CSI_FEATURES = NUM_CSI_TARGET_CHANNELS * NUM_CSI_SUBCARRIERS # 3 * 52 = 156 # *** CORRECTED TO 156 ***

# RSSI Config
NUM_RSSI_TARGET_CHANNELS = len(TARGET_CHANNELS)
NUM_RSSI_FEATURES = NUM_RSSI_TARGET_CHANNELS # 3

# Imputation for RSSI (Choose 'zero' or 'constant')
RSSI_IMPUTATION_METHOD = 'constant'
RSSI_MISSING_VALUE_FILL = -100.0 # Value for missing RSSI links/channels

# Ground Truth Mask Config
GT_MASK_SHAPE = (360, 360)
GT_MASK_DTYPE = np.uint8 # Or float64 if needed

# --- Helper Function: Parse CSI String (Keep as is) ---
def parse_csi_array_string(csi_str):
    """Parses the '[i,q,i,q,...]' string into a numpy array of complex numbers."""
    if not isinstance(csi_str, str) or not csi_str.startswith('[') or not csi_str.endswith(']'): return None
    csi_str = csi_str[1:-1]
    try:
        values = np.fromstring(csi_str, dtype=int, sep=',')
        if values.size != 128: return None # Expect HT20 size (64*2)
        complex_csi = values.astype(np.float32).view(np.complex64)
        return complex_csi
    except ValueError: return None
    except Exception as e: print(f"Error parsing CSI string: {e}"); traceback.print_exc(); return None

# --- Main Processing Function for One Recording ---
def process_recording_session(csi_csv_path, gt_npy_path, output_dir_csi, output_dir_rssi, is_calibration):
    """
    Processes a single recording session for BOTH CSI and RSSI features.
    Generates a zero mask if is_calibration is True.
    Returns True if processing and saving were successful for at least one feature type, False otherwise.
    """
    timestamp = csi_csv_path.stem.replace("csi_data_", "")
    print(f"  Processing {('Calibration' if is_calibration else 'Data')} Recording: {timestamp}")
    start_time = time.time()

    # --- Load Ground Truth (or create zero mask) ---
    ground_truth_mask = None
    if is_calibration:
        print(f"    Generating zero ground truth mask {GT_MASK_SHAPE}")
        ground_truth_mask = np.zeros(GT_MASK_SHAPE, dtype=GT_MASK_DTYPE)
    else:
        if not gt_npy_path or not gt_npy_path.is_file():
            print(f"    Error: Ground truth file missing for non-calibration data: {gt_npy_path}. Skipping.")
            return False
        try:
            ground_truth_mask = np.load(gt_npy_path)
            if ground_truth_mask.shape != GT_MASK_SHAPE:
                 print(f"    Warning: Ground truth mask {gt_npy_path.name} has unexpected shape {ground_truth_mask.shape}. Expected {GT_MASK_SHAPE}.")
        except Exception as e:
            print(f"    Error loading ground truth file {gt_npy_path}: {e}")
            traceback.print_exc()
            return False

    # --- Load CSV ---
    try:
        columns_to_load = ['sender_node', 'receiver_node', 'channel_index', 'rssi', 'csi_data_array']
        df = pd.read_csv(csi_csv_path, usecols=columns_to_load, low_memory=False)
        if df.empty: print(f"    Warning: CSV file is empty or missing required columns. Skipping."); return False
        essential_cols = ['sender_node', 'receiver_node', 'channel_index', 'rssi']
        for col in essential_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=essential_cols, inplace=True)
        if df.empty: print(f"    Warning: No valid metadata rows after type conversion. Skipping."); return False
        df = df.astype({'sender_node': int, 'receiver_node': int, 'channel_index': int, 'rssi': int})
    except FileNotFoundError: print(f"    Error: CSV file not found. Skipping."); return False
    except KeyError as e: print(f"    Error: Missing required column {e} in {csi_csv_path.name}. Skipping."); return False
    except Exception as e: print(f"    Error reading/parsing CSV file {csi_csv_path}: {e}"); traceback.print_exc(); return False

    # --- Filter Data ---
    df_filtered = df[
        df['channel_index'].isin(TARGET_CHANNELS) &
        (df['sender_node'] != df['receiver_node']) &
        (df['sender_node'] >= 0) & (df['sender_node'] < MAX_NODES) &
        (df['receiver_node'] >= 0) & (df['receiver_node'] < MAX_NODES)
    ].copy()
    if df_filtered.empty: print(f"    Warning: No valid, non-self-link data found for target channels {TARGET_CHANNELS}. Skipping file."); return False

    # ============================================
    # --- Process for RSSI Features (3, 16, 16) ---
    # ============================================
    print(f"    Processing RSSI ({len(df_filtered)} rows)...")
    feature_matrix_rssi = None
    try:
        averaged_rssi_series = df_filtered.groupby(['sender_node', 'receiver_node', 'channel_index'])['rssi'].mean()
        averaged_rssi = averaged_rssi_series.to_dict()
        if averaged_rssi:
            if RSSI_IMPUTATION_METHOD == 'constant':
                feature_matrix_rssi = np.full((NUM_RSSI_FEATURES, MAX_NODES, MAX_NODES), RSSI_MISSING_VALUE_FILL, dtype=np.float32)
            else: feature_matrix_rssi = np.zeros((NUM_RSSI_FEATURES, MAX_NODES, MAX_NODES), dtype=np.float32)
            channel_to_feature_idx_rssi = {ch: i for i, ch in enumerate(TARGET_CHANNELS)}
            for (sender, receiver, channel), avg_rssi in averaged_rssi.items():
                s_idx, r_idx = int(sender), int(receiver)
                if channel in channel_to_feature_idx_rssi:
                    feature_idx = channel_to_feature_idx_rssi[channel]
                    if 0 <= s_idx < MAX_NODES and 0 <= r_idx < MAX_NODES:
                        feature_matrix_rssi[feature_idx, s_idx, r_idx] = float(avg_rssi)
            if RSSI_IMPUTATION_METHOD == 'constant':
                for i in range(NUM_RSSI_FEATURES): np.fill_diagonal(feature_matrix_rssi[i, :, :], RSSI_MISSING_VALUE_FILL)
            print(f"      RSSI Matrix Construction Complete. Imputation: {RSSI_IMPUTATION_METHOD}.")
        else: print("      Warning: No averaged RSSI data to create matrix.")
    except Exception as e: print(f"    Error during RSSI processing: {e}"); traceback.print_exc(); feature_matrix_rssi = None

    # ============================================
    # --- Process for CSI Features (156, 16, 16) -
    # ============================================
    print(f"    Processing CSI ({len(df_filtered)} rows)...")
    feature_matrix_csi = None
    try:
        grouped_csi_amps = defaultdict(list)
        rows_processed_csi = 0
        rows_skipped_csi = 0
        required_csi_len = max(DATA_SUBCARRIER_INDICES) + 1 if DATA_SUBCARRIER_INDICES else 0

        for index, row in df_filtered.iterrows():
            try:
                complex_csi = parse_csi_array_string(row['csi_data_array'])
                if complex_csi is None: rows_skipped_csi += 1; continue
                if len(complex_csi) < required_csi_len: rows_skipped_csi += 1; continue
                selected_complex = complex_csi[DATA_SUBCARRIER_INDICES] # Selects 52 values
                amplitudes = np.abs(selected_complex).astype(np.float32) # Shape (52,)
                key = (int(row['sender_node']), int(row['receiver_node']), int(row['channel_index']))
                grouped_csi_amps[key].append(amplitudes)
                rows_processed_csi += 1
            except Exception as e: print(f"    Error processing CSI for row {index}: {e}"); rows_skipped_csi += 1

        if rows_skipped_csi > 0: print(f"      Note: Skipped {rows_skipped_csi} rows during CSI parsing/selection.")

        averaged_csi_amps = {}
        if grouped_csi_amps:
            for key, amp_list in grouped_csi_amps.items():
                if amp_list:
                    try:
                        stacked_amps = np.stack(amp_list, axis=0)
                        mean_amps = np.mean(stacked_amps, axis=0)
                        if mean_amps.shape == (NUM_CSI_SUBCARRIERS,): # Check shape is (52,)
                            averaged_csi_amps[key] = mean_amps.astype(np.float32)
                        else: print(f"      Warning: CSI Avg amp array key {key} shape {mean_amps.shape} != ({NUM_CSI_SUBCARRIERS},).")
                    except ValueError as e: print(f"      Warning: Error stacking/avg CSI key {key}: {e}. List len: {len(amp_list)}")
                    except Exception as e: print(f"      Warning: Unexpected error avg CSI key {key}: {e}")

        if averaged_csi_amps:
            # Construct CSI Feature Matrix (using NUM_FEATURES=156, NUM_SUBCARRIERS=52)
            feature_matrix_csi = np.zeros((NUM_CSI_FEATURES, MAX_NODES, MAX_NODES), dtype=np.float32) # Shape (156, 16, 16)
            channel_to_offset_csi = {ch: i * NUM_CSI_SUBCARRIERS for i, ch in enumerate(TARGET_CHANNELS)} # Offsets 0, 52, 104

            for (sender, receiver, channel), avg_amps in averaged_csi_amps.items():
                if channel in channel_to_offset_csi:
                    offset = channel_to_offset_csi[channel]
                    s_idx, r_idx = int(sender), int(receiver)
                    if 0 <= s_idx < MAX_NODES and 0 <= r_idx < MAX_NODES:
                         if avg_amps.shape == (NUM_CSI_SUBCARRIERS,): # Check shape is (52,)
                             feature_matrix_csi[offset : offset + NUM_CSI_SUBCARRIERS, s_idx, r_idx] = avg_amps
                         # else: pass # Already warned
                    # else: pass # Already warned
            print(f"      CSI Matrix Construction Complete. Imputation: Zero Filling.")
        else:
            print("      Warning: No averaged CSI data to create matrix.")

    except Exception as e:
        print(f"    Error during CSI processing: {e}")
        traceback.print_exc()
        feature_matrix_csi = None # Mark as failed

    # --- Phase 4: Saving ---
    save_success_rssi = False
    save_success_csi = False
    save_success_gt = False

    try:
        # Save RSSI Data if available
        if feature_matrix_rssi is not None:
            output_dir_rssi.mkdir(parents=True, exist_ok=True)
            output_rssi_path = output_dir_rssi / "rssi_features.npy"
            output_gt_path_rssi = output_dir_rssi / "ground_truth.npy"
            np.save(output_rssi_path, feature_matrix_rssi)
            np.save(output_gt_path_rssi, ground_truth_mask)
            print(f"      Saved RSSI features: {output_rssi_path}")
            print(f"      Saved GT mask to RSSI dir: {output_gt_path_rssi}")
            save_success_rssi = True
            save_success_gt = True
        else: print("      Skipping RSSI save (no data).")

        # Save CSI Data if available
        if feature_matrix_csi is not None:
            output_dir_csi.mkdir(parents=True, exist_ok=True)
            output_csi_path = output_dir_csi / "csi_features.npy"
            output_gt_path_csi = output_dir_csi / "ground_truth.npy"
            np.save(output_csi_path, feature_matrix_csi)
            if not save_success_gt: # Only save GT if not already saved with RSSI
                 np.save(output_gt_path_csi, ground_truth_mask)
                 print(f"      Saved GT mask to CSI dir: {output_gt_path_csi}")
            else: # Otherwise, copy for consistency
                 try: shutil.copy2(output_gt_path_rssi, output_gt_path_csi); print(f"      Copied GT mask to CSI dir: {output_gt_path_csi}")
                 except Exception as copy_e: print(f"      Warning: Failed to copy GT mask to CSI dir: {copy_e}")
            print(f"      Saved CSI features: {output_csi_path}")
            save_success_csi = True
        else: print("      Skipping CSI save (no data).")

        processing_time = time.time() - start_time
        print(f"    Finished processing in {processing_time:.2f} seconds.")
        return save_success_rssi or save_success_csi

    except Exception as e:
        print(f"    Error saving processed files for {timestamp}: {e}")
        traceback.print_exc()
        return False


# --- Main Orchestration Function ---
def process_dataset(raw_data_dir, processed_dir_csi, processed_dir_rssi):
    """Finds matching files and processes each recording session for RSSI & CSI."""
    print(f"Starting data processing...")
    print(f"Raw data source: {raw_data_dir}")
    print(f"Processed CSI destination: {processed_dir_csi}")
    print(f"Processed RSSI destination: {processed_dir_rssi}")

    raw_data_path = Path(raw_data_dir).resolve()
    processed_csi_path = Path(processed_dir_csi).resolve()
    processed_rssi_path = Path(processed_dir_rssi).resolve()

    if not raw_data_path.is_dir(): print(f"Error: Raw data directory not found: {raw_data_path}"); return

    dataset_dirs = sorted([d for d in raw_data_path.iterdir() if d.is_dir() and \
                           (re.match(r'^potato_\d+$', d.name) or re.match(r'^calibration(_\d+)?$', d.name))])
    if not dataset_dirs: print(f"Warning: No 'potato_X' or 'calibration_X' directories found in {raw_data_path}"); return

    total_processed = 0
    total_skipped_missing = 0
    total_errors_processing = 0
    overall_start_time = time.time()

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        is_calibration = dataset_name.startswith("calibration")
        gt_dir_path = None

        print(f"\nProcessing dataset: {dataset_name} {'(Calibration)' if is_calibration else ''}")

        if not is_calibration:
            gt_dir_name = f"{dataset_name}_gt"
            gt_dir_path = raw_data_path / gt_dir_name
            if not gt_dir_path.is_dir():
                print(f"  Warning: Ground truth directory '{gt_dir_path}' not found, skipping dataset.")
                continue
        # else: gt_dir_path remains None for calibration

        recording_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith('rec_')])
        if not recording_dirs: print(f"  Warning: No 'rec_...' subdirectories found in {dataset_dir}"); continue

        print(f"  Found {len(recording_dirs)} recording directories.")

        for rec_dir in recording_dirs:
            match = re.match(r'^rec_(\d{8}_\d{6})$', rec_dir.name)
            if not match: continue

            timestamp = match.group(1)
            csi_file_name = f"csi_data_{timestamp}.csv"
            src_csi_path = rec_dir / csi_file_name
            src_gt_path = None # Default to None

            if not src_csi_path.is_file():
                print(f"    Skipping {timestamp}: CSI file missing ({src_csi_path.name})")
                total_skipped_missing += 1
                continue

            if not is_calibration:
                gt_file_name = f"{timestamp}.npy"
                src_gt_path = gt_dir_path / gt_file_name
                if not src_gt_path.is_file():
                    print(f"    Skipping {timestamp}: Ground truth file missing ({src_gt_path.name})")
                    total_skipped_missing += 1
                    continue

            # Define output directories
            output_dataset_name = "calibration" if is_calibration else dataset_name
            output_rec_dir_csi = processed_csi_path / output_dataset_name / timestamp
            output_rec_dir_rssi = processed_rssi_path / output_dataset_name / timestamp

            # Process this recording session
            success = process_recording_session(
                src_csi_path,
                src_gt_path, # Will be None for calibration
                output_rec_dir_csi,
                output_rec_dir_rssi,
                is_calibration
            )

            if success:
                total_processed += 1
            else:
                total_errors_processing += 1

    overall_end_time = time.time()
    print("\n------------------------------------")
    print("Data processing complete.")
    print(f"Total time: {overall_end_time - overall_start_time:.2f} seconds")
    print(f"Total recordings processed successfully (CSI and/or RSSI): {total_processed}")
    print(f"Total recordings skipped (missing files): {total_skipped_missing}")
    print(f"Total recordings with errors during processing: {total_errors_processing}")
    print("------------------------------------")


# --- Script Entry Point ---
if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    default_raw_data = project_root / "raw_data"
    default_processed_csi = project_root / "processed_data_csi"
    default_processed_rssi = project_root / "processed_data_rssi"

    parser = argparse.ArgumentParser(description="Process raw CSI CSV data into CSI and RSSI feature matrices.")
    parser.add_argument("--raw_dir", type=str, default=str(default_raw_data), help=f"Path to raw_data directory (default: {default_raw_data})")
    parser.add_argument("--processed_dir_csi", type=str, default=str(default_processed_csi), help=f"Path to processed_data_csi directory (default: {default_processed_csi})")
    parser.add_argument("--processed_dir_rssi", type=str, default=str(default_processed_rssi), help=f"Path to processed_data_rssi directory (default: {default_processed_rssi})")
    args = parser.parse_args()

    Path(args.processed_dir_csi).mkdir(parents=True, exist_ok=True)
    Path(args.processed_dir_rssi).mkdir(parents=True, exist_ok=True)

    process_dataset(args.raw_dir, args.processed_dir_csi, args.processed_dir_rssi)