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

# !!=========================================================================!!
# !! Indices for the 52 commonly used DATA subcarriers in HT20               !!
# !! Assumes the 64 complex values correspond to subcarriers -32 to +31.     !!
# !! Excludes DC (0) and standard pilots (-21, -7, +7, +21).                 !!
# !!=========================================================================!!
_HT20_PILOT_OFFSETS = {-21, -7, 7, 21}
_HT20_NULL_DC_OFFSET = {0}
# Generate offsets -28 to +28 (inclusive)
all_potential_data_pilot_offsets = set(range(-28, 29))
# Remove DC and Pilots
data_relative_offsets = all_potential_data_pilot_offsets - _HT20_NULL_DC_OFFSET - _HT20_PILOT_OFFSETS
# Map to 0-63 array indices (assuming index 32 is DC)
center_index_in_64 = 32
DATA_SUBCARRIER_INDICES = sorted(
    [center_index_in_64 + offset for offset in data_relative_offsets]
)
# --- >>>>>>>> VERIFICATION PASSED (Generates 52 indices) <<<<<<<< ---

# --- Sanity Check ---
if len(DATA_SUBCARRIER_INDICES) != 52: # *** EXPECT 52 ***
    print(f"ERROR: DATA_SUBCARRIER_INDICES length is {len(DATA_SUBCARRIER_INDICES)}, expected 52. Check definition.")
    print(f"Generated Indices ({len(DATA_SUBCARRIER_INDICES)}): {DATA_SUBCARRIER_INDICES}")
    exit(1)
else:
    print(f"Using {len(DATA_SUBCARRIER_INDICES)} subcarrier indices.")
    # print(f"Indices: {DATA_SUBCARRIER_INDICES}")
# --------------------

TARGET_CHANNELS = [1, 6, 11]
MAX_NODES = 16
NUM_SUBCARRIERS = 52       # *** SET TO 52 ***
NUM_TARGET_CHANNELS = len(TARGET_CHANNELS)
NUM_FEATURES = NUM_TARGET_CHANNELS * NUM_SUBCARRIERS # 3 * 52 = 156 # *** SET TO 156 ***

# --- Helper Function: Parse CSI String ---
def parse_csi_array_string(csi_str):
    """Parses the '[i,q,i,q,...]' string into a numpy array of complex numbers."""
    if not isinstance(csi_str, str) or not csi_str.startswith('[') or not csi_str.endswith(']'):
        return None
    csi_str = csi_str[1:-1]
    try:
        values = np.fromstring(csi_str, dtype=int, sep=',')
        # Expecting 128 values for 64 complex numbers (HT-LTF 20MHz)
        if values.size != 128:
            # print(f"Warning: Unexpected number of values ({values.size}) in CSI string. Expected 128.")
            return None # Or handle differently if other lengths are valid
        # Reshape and create complex numbers
        complex_csi = values.astype(np.float32).view(np.complex64)
        return complex_csi
    except ValueError:
        return None
    except Exception as e:
        print(f"Error parsing CSI string: {e}")
        traceback.print_exc()
        return None

# --- Main Processing Function for One Recording ---
def process_recording_session(csi_csv_path, gt_npy_path, output_dir):
    """Processes a single recording session (CSV + NPY)"""
    print(f"  Processing: {csi_csv_path.name}")
    timestamp = csi_csv_path.stem.replace("csi_data_", "")

    # --- Phase 1: Load CSV and Extract Amplitudes ---
    try:
        df = pd.read_csv(csi_csv_path, low_memory=False)
        if df.empty: print(f"    Warning: CSV file is empty. Skipping."); return False
    except FileNotFoundError: print(f"    Error: CSV file not found. Skipping."); return False
    except Exception as e: print(f"    Error reading CSV file {csi_csv_path}: {e}"); traceback.print_exc(); return False

    all_measurements = []
    skipped_rows = 0
    required_csi_len = max(DATA_SUBCARRIER_INDICES) + 1 if DATA_SUBCARRIER_INDICES else 0

    for index, row in df.iterrows():
        try:
            complex_csi = parse_csi_array_string(row['csi_data_array'])
            if complex_csi is None: skipped_rows += 1; continue

            if len(complex_csi) < required_csi_len: skipped_rows += 1; continue
            selected_complex = complex_csi[DATA_SUBCARRIER_INDICES] # Selects 52 values

            amplitudes = np.abs(selected_complex).astype(np.float32) # Shape (52,)

            all_measurements.append({
                "sender": int(row['sender_node']),
                "receiver": int(row['receiver_node']),
                "channel": int(row['channel_index']),
                "amplitudes": amplitudes # The 52 amplitude values
            })
        except KeyError as e: print(f"    Warning: Missing column '{e}' row {index}. Skip."); skipped_rows += 1
        except Exception as e: print(f"    Error processing row {index}: {e}"); traceback.print_exc(); skipped_rows += 1

    if skipped_rows > 0: print(f"    Note: Skipped {skipped_rows} rows during processing.")
    if not all_measurements: print(f"    Warning: No valid CSI measurements found. Skipping file."); return False

    # --- Phase 2: Aggregate (Average) and Format ---
    grouped_data = defaultdict(list)
    for meas in all_measurements:
        if meas['sender'] == meas['receiver']: continue
        if meas['channel'] not in TARGET_CHANNELS: continue
        key = (meas['sender'], meas['receiver'], meas['channel'])
        grouped_data[key].append(meas['amplitudes'])

    averaged_csi = {}
    for key, amp_list in grouped_data.items():
        if amp_list:
             try:
                 stacked_amps = np.stack(amp_list, axis=0)
                 mean_amps = np.mean(stacked_amps, axis=0)
                 if mean_amps.shape == (NUM_SUBCARRIERS,): # Check shape is (52,)
                     averaged_csi[key] = mean_amps.astype(np.float32)
                 else: print(f"    Warning: Avg amp array key {key} shape {mean_amps.shape} != ({NUM_SUBCARRIERS},).")
             except ValueError as e: print(f"    Warning: Error stacking/avg key {key}: {e}. List len: {len(amp_list)}")
             except Exception as e: print(f"    Warning: Unexpected error avg key {key}: {e}")

    # Construct Feature Matrix (using NUM_FEATURES=156, NUM_SUBCARRIERS=52)
    feature_matrix = np.zeros((NUM_FEATURES, MAX_NODES, MAX_NODES), dtype=np.float32) # Shape (156, 16, 16)
    channel_to_offset = {ch: i * NUM_SUBCARRIERS for i, ch in enumerate(TARGET_CHANNELS)} # Offsets 0, 52, 104

    for (sender, receiver, channel), avg_amps in averaged_csi.items():
        if channel in channel_to_offset:
            offset = channel_to_offset[channel]
            if 0 <= sender < MAX_NODES and 0 <= receiver < MAX_NODES:
                 if avg_amps.shape == (NUM_SUBCARRIERS,): # Check shape is (52,)
                     feature_matrix[offset : offset + NUM_SUBCARRIERS, sender, receiver] = avg_amps
                 # else: # Already warned
                 #      pass
            # else: # Should be caught earlier
            #      pass

    # --- Phase 3: Imputation (Zero Filling - Default) ---
    print(f"    Using Zero Filling for missing data.")

    # --- Phase 4: Saving ---
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csi_path = output_dir / "csi_features.npy"
        output_gt_path = output_dir / "ground_truth.npy"
        np.save(output_csi_path, feature_matrix)
        print(f"      Saved features: {output_csi_path}")
        shutil.copy2(gt_npy_path, output_gt_path)
        print(f"      Copied ground truth: {output_gt_path}")
        return True
    except Exception as e:
        print(f"    Error saving processed files for {timestamp}: {e}")
        traceback.print_exc()
        return False


# --- Main Orchestration Function ---
def process_dataset(raw_data_dir, processed_data_dir):
    """Finds matching files and processes each recording session."""
    print(f"Starting data processing...")
    print(f"Raw data source: {raw_data_dir}")
    print(f"Processed data destination: {processed_data_dir}")

    raw_data_path = Path(raw_data_dir).resolve()
    processed_data_path = Path(processed_data_dir).resolve()

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

            # Define output directory for this recording
            output_rec_dir = processed_data_path / dataset_name / timestamp

            # Process this recording session
            success = process_recording_session(src_csi_path, src_gt_path, output_rec_dir)

            if success:
                total_processed += 1
            else:
                total_errors_processing += 1

    print("\n------------------------------------")
    print("Data processing complete.")
    print(f"Total recordings processed successfully: {total_processed}")
    print(f"Total recordings skipped (missing files): {total_skipped_missing}")
    print(f"Total recordings with errors during processing: {total_errors_processing}")
    print("------------------------------------")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    default_raw_data = project_root / "raw_data"
    default_processed_data = project_root / "processed_data_csi"

    parser = argparse.ArgumentParser(description="Process raw CSI CSV data into feature matrices.")
    parser.add_argument("--raw_dir", type=str, default=str(default_raw_data), help=f"Path to raw_data directory (default: {default_raw_data})")
    parser.add_argument("--processed_dir", type=str, default=str(default_processed_data), help=f"Path to processed_data directory (default: {default_processed_data})")
    args = parser.parse_args()

    Path(args.processed_dir).mkdir(parents=True, exist_ok=True)
    process_dataset(args.raw_dir, args.processed_dir)