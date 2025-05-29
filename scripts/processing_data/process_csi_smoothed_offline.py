# scripts/process_csi_smoothed_offline.py

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
import time
from scipy.signal import savgol_filter # For Savitzky-Golay smoothing
from scipy.ndimage import gaussian_filter1d # For Gaussian smoothing

# --- Configuration ---

# !! Indices for the 52 commonly used DATA subcarriers in HT20 !!
# (Copied from process_all_data.py - ensure consistency)
_HT20_PILOT_OFFSETS = {-21, -7, 7, 21}
_HT20_NULL_DC_OFFSET = {0}
_all_potential_data_pilot_offsets = set(range(-28, 29))
_data_relative_offsets = _all_potential_data_pilot_offsets - _HT20_NULL_DC_OFFSET - _HT20_PILOT_OFFSETS
_center_index_in_64 = 32
DATA_SUBCARRIER_INDICES = sorted([_center_index_in_64 + offset for offset in _data_relative_offsets])
NUM_CSI_SUBCARRIERS = len(DATA_SUBCARRIER_INDICES)

if NUM_CSI_SUBCARRIERS != 52:
    print(f"ERROR: DATA_SUBCARRIER_INDICES length is {NUM_CSI_SUBCARRIERS}, expected 52.")
    exit(1)
else:
    print(f"Using {NUM_CSI_SUBCARRIERS} subcarrier indices for CSI processing.")
# --------------------

TARGET_CHANNELS = [1, 6, 11]
MAX_NODES = 16
NUM_CSI_TARGET_CHANNELS = len(TARGET_CHANNELS)
NUM_CSI_FEATURES = NUM_CSI_TARGET_CHANNELS * NUM_CSI_SUBCARRIERS # 156

# Ground Truth Mask Config
GT_MASK_SHAPE = (360, 360)
GT_MASK_DTYPE = np.uint8

# --- Helper Function: Parse CSI String ---
# (Copied from process_all_data.py - ensure consistency)
def parse_csi_array_string(csi_str):
    if not isinstance(csi_str, str) or not csi_str.startswith('[') or not csi_str.endswith(']'): return None
    csi_str = csi_str[1:-1]
    try:
        values = np.fromstring(csi_str, dtype=int, sep=',')
        if values.size != 128: return None
        complex_csi = values.astype(np.float32).view(np.complex64)
        if len(complex_csi) <= max(DATA_SUBCARRIER_INDICES): return None
        return complex_csi
    except Exception: return None

# --- Smoothing Function ---
# (Adapted from plotting script)
def apply_smoothing(data_matrix, method='none', window=5, polyorder=2, sigma=1.0):
    """Applies smoothing along the time axis (axis 0) of the data matrix."""
    if method == 'none' or data_matrix.shape[0] < 2 or data_matrix.shape[0] < window:
        # print(f"Smoothing: None (method={method}, packets={data_matrix.shape[0]}, window={window})")
        return data_matrix

    smoothed_matrix = np.zeros_like(data_matrix, dtype=np.float32)

    if method == 'sma':
        # print(f"Smoothing: SMA (window={window})")
        df = pd.DataFrame(data_matrix)
        smoothed_matrix = df.rolling(window=window, min_periods=1, center=False).mean().to_numpy(dtype=np.float32)
    elif method == 'ema':
        # print(f"Smoothing: EMA (span={window})")
        df = pd.DataFrame(data_matrix)
        smoothed_matrix = df.ewm(span=window, adjust=False, min_periods=1).mean().to_numpy(dtype=np.float32)
    elif method == 'gaussian':
        # print(f"Smoothing: Gaussian (sigma={sigma})")
        for i in range(data_matrix.shape[1]):
            smoothed_matrix[:, i] = gaussian_filter1d(data_matrix[:, i], sigma=sigma, mode='nearest')
    elif method == 'savgol':
        if window % 2 == 0: window += 1
        window = max(window, polyorder + 1 + (polyorder % 2))
        # print(f"Smoothing: Savgol (window={window}, polyorder={polyorder})")
        if data_matrix.shape[0] < window:
             # print(f"Warning: Not enough data points ({data_matrix.shape[0]}) for Savgol window {window}. Skipping smoothing.")
             return data_matrix
        try:
            smoothed_matrix = savgol_filter(data_matrix, window_length=window, polyorder=polyorder, axis=0, mode='interp').astype(np.float32)
        except ValueError as e:
             print(f"Warning: Error during Savgol filtering: {e}. Skipping smoothing.")
             return data_matrix
    else:
        # print(f"Warning: Unknown smoothing method '{method}'. No smoothing applied.")
        return data_matrix

    return smoothed_matrix

# --- Main Processing Function for One Recording ---
def process_smoothed_session(
    raw_csv_path: Path,
    gt_npy_path: Path | None, # Path to ground truth .npy or None for calibration
    output_dir_csi_smooth: Path,
    is_calibration: bool,
    smoothing_method: str,
    smoothing_window: int,
    smoothing_polyorder: int,
    smoothing_sigma: float
):
    """
    Processes a single recording session: reads raw CSV, smooths CSI per link/channel
    across packets, averages the smoothed data, and saves the result.
    """
    timestamp = raw_csv_path.stem.replace("csi_data_", "")
    print(f"  Processing {('Calibration' if is_calibration else 'Data')} Recording: {timestamp} (Smoothing: {smoothing_method}, Win={smoothing_window})")
    start_time = time.time()

    # --- Load Ground Truth (or create zero mask) ---
    ground_truth_mask = None
    gt_load_success = False
    if is_calibration:
        print(f"    Generating zero ground truth mask {GT_MASK_SHAPE}")
        ground_truth_mask = np.zeros(GT_MASK_SHAPE, dtype=GT_MASK_DTYPE)
        gt_load_success = True
    else:
        if not gt_npy_path or not gt_npy_path.is_file():
            print(f"    Error: Ground truth file missing for non-calibration data: {gt_npy_path}. Skipping session.")
            return False
        try:
            ground_truth_mask = np.load(gt_npy_path)
            if ground_truth_mask.shape != GT_MASK_SHAPE:
                 print(f"    Warning: Ground truth mask {gt_npy_path.name} has unexpected shape {ground_truth_mask.shape}. Expected {GT_MASK_SHAPE}.")
                 # Continue anyway, but warn
            gt_load_success = True
        except Exception as e:
            print(f"    Error loading ground truth file {gt_npy_path}: {e}")
            traceback.print_exc()
            return False # Skip session if GT fails for non-calibration

    # --- Load Raw CSV ---
    try:
        columns_to_load = ['sender_node', 'receiver_node', 'channel_index', 'csi_data_array']
        df = pd.read_csv(raw_csv_path, usecols=columns_to_load, low_memory=False)
        if df.empty: print(f"    Warning: Raw CSV file {raw_csv_path.name} is empty. Skipping."); return False

        # Convert types and drop invalid rows early
        for col in ['sender_node', 'receiver_node', 'channel_index']:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['sender_node', 'receiver_node', 'channel_index'], inplace=True)
        if df.empty: print(f"    Warning: No valid metadata rows after type conversion. Skipping."); return False
        df = df.astype({'sender_node': int, 'receiver_node': int, 'channel_index': int})

    except FileNotFoundError: print(f"    Error: Raw CSV file not found: {raw_csv_path}. Skipping."); return False
    except KeyError as e: print(f"    Error: Missing required column {e} in {raw_csv_path.name}. Skipping."); return False
    except Exception as e: print(f"    Error reading/parsing CSV file {raw_csv_path}: {e}"); traceback.print_exc(); return False

    # --- Process per Link/Channel, Smooth, then Average ---
    feature_matrix_csi_smooth = np.zeros((NUM_CSI_FEATURES, MAX_NODES, MAX_NODES), dtype=np.float32)
    channel_to_offset_csi = {ch: i * NUM_CSI_SUBCARRIERS for i, ch in enumerate(TARGET_CHANNELS)}
    links_processed = 0
    links_skipped_no_packets = 0
    links_skipped_no_csi = 0

    required_csi_len = max(DATA_SUBCARRIER_INDICES) + 1

    for sender in range(MAX_NODES):
        for receiver in range(MAX_NODES):
            if sender == receiver: continue # Skip self-links

            for channel in TARGET_CHANNELS:
                # Filter DataFrame for this specific link/channel
                df_link_channel = df[
                    (df['sender_node'] == sender) &
                    (df['receiver_node'] == receiver) &
                    (df['channel_index'] == channel)
                ]

                if df_link_channel.empty:
                    links_skipped_no_packets += 1
                    continue # Skip if no packets found for this link/channel

                # Extract CSI amplitudes for all packets of this link/channel
                packet_amplitudes_list = []
                for index, row in df_link_channel.iterrows():
                    complex_csi = parse_csi_array_string(row['csi_data_array'])
                    if complex_csi is None or len(complex_csi) < required_csi_len: continue
                    try:
                        selected_complex = complex_csi[DATA_SUBCARRIER_INDICES]
                        amplitudes = np.abs(selected_complex).astype(np.float32)
                        if amplitudes.shape == (NUM_CSI_SUBCARRIERS,):
                            packet_amplitudes_list.append(amplitudes)
                    except Exception:
                        continue # Skip packet on error

                if not packet_amplitudes_list:
                    links_skipped_no_csi += 1
                    continue # Skip if no valid CSI extracted for this link/channel

                # Stack into matrix: (num_packets, 52)
                amplitude_matrix = np.stack(packet_amplitudes_list, axis=0)

                # Apply smoothing across packets (axis 0)
                smoothed_amplitude_matrix = apply_smoothing(
                    amplitude_matrix,
                    method=smoothing_method,
                    window=smoothing_window,
                    polyorder=smoothing_polyorder,
                    sigma=smoothing_sigma
                )

                # Average the smoothed amplitudes across packets
                avg_smoothed_amps = np.mean(smoothed_amplitude_matrix, axis=0).astype(np.float32) # Shape (52,)

                # Place into the final feature matrix
                if avg_smoothed_amps.shape == (NUM_CSI_SUBCARRIERS,):
                    offset = channel_to_offset_csi[channel]
                    feature_matrix_csi_smooth[offset : offset + NUM_CSI_SUBCARRIERS, sender, receiver] = avg_smoothed_amps
                    links_processed += 1
                else:
                    print(f"    Warning: Avg smoothed amp shape mismatch for link {sender}->{receiver} ch {channel}. Shape: {avg_smoothed_amps.shape}")


    print(f"    Processed {links_processed} links. Skipped {links_skipped_no_packets} (no packets), {links_skipped_no_csi} (no valid CSI).")

    # --- Saving ---
    save_success = False
    try:
        output_dir_csi_smooth.mkdir(parents=True, exist_ok=True)
        output_csi_path = output_dir_csi_smooth / "csi_features.npy"
        output_gt_path = output_dir_csi_smooth / "ground_truth.npy"

        np.save(output_csi_path, feature_matrix_csi_smooth)
        print(f"      Saved SMOOTHED CSI features: {output_csi_path}")

        if gt_load_success:
            np.save(output_gt_path, ground_truth_mask)
            print(f"      Saved GT mask: {output_gt_path}")
        else:
             # This case should ideally not be reached for non-calibration if GT load fails
             print(f"      Warning: GT mask was not loaded successfully, not saving GT file.")


        save_success = True

    except Exception as e:
        print(f"    Error saving processed files for {timestamp}: {e}")
        traceback.print_exc()

    processing_time = time.time() - start_time
    print(f"    Finished processing in {processing_time:.2f} seconds.")
    return save_success


# --- Main Orchestration Function ---
def process_smoothed_dataset(
    raw_data_dir: str,
    processed_dir_csi_smooth: str,
    smoothing_method: str,
    smoothing_window: int,
    smoothing_polyorder: int,
    smoothing_sigma: float
):
    """Finds raw CSVs, matches GTs, and processes each session with smoothing."""
    print(f"Starting OFFLINE SMOOTHED CSI data processing...")
    print(f"Raw data source: {raw_data_dir}")
    print(f"Smoothed CSI destination: {processed_dir_csi_smooth}")
    print(f"Smoothing: method={smoothing_method}, window={smoothing_window}, polyorder={smoothing_polyorder}, sigma={smoothing_sigma}")

    raw_data_path = Path(raw_data_dir).resolve()
    processed_csi_smooth_path = Path(processed_dir_csi_smooth).resolve()

    if not raw_data_path.is_dir(): print(f"Error: Raw data directory not found: {raw_data_path}"); return

    # Find dataset directories (potato_X, calibration)
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
        gt_dir_path = None # Path to the _gt directory

        print(f"\nProcessing dataset: {dataset_name} {'(Calibration)' if is_calibration else ''}")

        # Find corresponding ground truth directory if not calibration
        if not is_calibration:
            gt_dir_name = f"{dataset_name}_gt"
            gt_dir_path = raw_data_path / gt_dir_name
            if not gt_dir_path.is_dir():
                print(f"  Warning: Ground truth directory '{gt_dir_path}' not found, skipping dataset.")
                continue

        # Find recording directories (rec_...) within the dataset directory
        recording_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith('rec_')])
        if not recording_dirs: print(f"  Warning: No 'rec_...' subdirectories found in {dataset_dir}"); continue

        print(f"  Found {len(recording_dirs)} recording directories.")

        for rec_dir in recording_dirs:
            match = re.match(r'^rec_(\d{8}_\d{6})$', rec_dir.name)
            if not match: continue

            timestamp = match.group(1)
            csi_file_name = f"csi_data_{timestamp}.csv"
            src_csv_path = rec_dir / csi_file_name # Path to the RAW CSV
            src_gt_path = None # Default to None

            # Check if RAW CSV exists
            if not src_csv_path.is_file():
                print(f"    Skipping {timestamp}: Raw CSI CSV file missing ({src_csv_path.name})")
                total_skipped_missing += 1
                continue

            # Find corresponding GT file if not calibration
            if not is_calibration:
                gt_file_name = f"{timestamp}.npy"
                src_gt_path = gt_dir_path / gt_file_name
                if not src_gt_path.is_file():
                    print(f"    Skipping {timestamp}: Ground truth file missing ({src_gt_path.name})")
                    total_skipped_missing += 1
                    continue

            # Define output directory for this session's smoothed data
            output_dataset_name = "calibration" if is_calibration else dataset_name
            output_rec_dir_csi_smooth = processed_csi_smooth_path / output_dataset_name / timestamp

            # Process this recording session with smoothing
            success = process_smoothed_session(
                src_csv_path,
                src_gt_path, # Will be None for calibration
                output_rec_dir_csi_smooth,
                is_calibration,
                smoothing_method,
                smoothing_window,
                smoothing_polyorder,
                smoothing_sigma
            )

            if success:
                total_processed += 1
            else:
                total_errors_processing += 1

    overall_end_time = time.time()
    print("\n------------------------------------")
    print("Offline Smoothed CSI data processing complete.")
    print(f"Total time: {overall_end_time - overall_start_time:.2f} seconds")
    print(f"Total recordings processed successfully: {total_processed}")
    print(f"Total recordings skipped (missing files): {total_skipped_missing}")
    print(f"Total recordings with errors during processing: {total_errors_processing}")
    print("------------------------------------")


# --- Script Entry Point ---
if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    default_raw_data = project_root / "raw_data"
    default_processed_csi_smooth = project_root / "processed_data_csi_smooth" # New default output

    parser = argparse.ArgumentParser(description="Process raw CSI CSV data into SMOOTHED CSI feature matrices (offline, packet-level smoothing).")
    parser.add_argument("--raw_dir", type=str, default=str(default_raw_data),
                        help=f"Path to raw_data directory (default: {default_raw_data})")
    parser.add_argument("--processed_dir_csi_smooth", type=str, default=str(default_processed_csi_smooth),
                        help=f"Path to save smoothed processed CSI data (default: {default_processed_csi_smooth})")
    # Smoothing arguments
    parser.add_argument("--smooth", type=str, default='none', choices=['none', 'sma', 'ema', 'gaussian', 'savgol'],
                        help="Smoothing method to apply across packets per link (default: none).")
    parser.add_argument("--smooth_window", type=int, default=5,
                        help="Window size for SMA/Savgol, approx span for EMA (default: 5).")
    parser.add_argument("--smooth_polyorder", type=int, default=2,
                        help="Polynomial order for Savgol filter (default: 2).")
    parser.add_argument("--smooth_sigma", type=float, default=1.0,
                        help="Sigma for Gaussian filter (default: 1.0).")

    args = parser.parse_args()

    # Create the base output directory
    Path(args.processed_dir_csi_smooth).mkdir(parents=True, exist_ok=True)

    process_smoothed_dataset(
        args.raw_dir,
        args.processed_dir_csi_smooth,
        args.smooth,
        args.smooth_window,
        args.smooth_polyorder,
        args.smooth_sigma
        )