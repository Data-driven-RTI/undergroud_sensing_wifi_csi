import os
import shutil
import re
import argparse
from pathlib import Path

def organize_data(raw_data_dir, processed_data_dir):
    """
    Organizes raw CSI, ground truth, and image data into a processed structure.

    Args:
        raw_data_dir (str): Path to the main 'raw_data' directory.
        processed_data_dir (str): Path to the main 'processed_data' directory.
    """
    print(f"Starting data organization...")
    print(f"Raw data source: {raw_data_dir}")
    print(f"Processed data destination: {processed_data_dir}")

    raw_data_path = Path(raw_data_dir).resolve()
    processed_data_path = Path(processed_data_dir).resolve()

    if not raw_data_path.is_dir():
        print(f"Error: Raw data directory not found: {raw_data_path}")
        return

    # Ensure the base processed_data directory exists
    processed_data_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured processed data directory exists: {processed_data_path}")

    # Find potato_X directories
    potato_dirs = sorted([d for d in raw_data_path.iterdir() if d.is_dir() and re.match(r'^potato_\d+$', d.name)])

    if not potato_dirs:
        print(f"Warning: No 'potato_X' directories found in {raw_data_path}")
        return

    total_recordings_processed = 0
    total_recordings_skipped = 0

    for potato_dir in potato_dirs:
        dataset_name = potato_dir.name # e.g., "potato_1"
        gt_dir_name = f"{dataset_name}_gt"
        gt_dir_path = raw_data_path / gt_dir_name

        print(f"\nProcessing dataset: {dataset_name}")

        # Check if corresponding ground truth directory exists
        if not gt_dir_path.is_dir():
            print(f"  Warning: Ground truth directory not found, skipping dataset: {gt_dir_path}")
            continue

        # Create corresponding output directory in processed_data
        output_dataset_path = processed_data_path / dataset_name
        output_dataset_path.mkdir(parents=True, exist_ok=True)
        print(f"  Output directory: {output_dataset_path}")

        # Find rec_YYYYMMDD_HHMMSS directories inside the potato_X directory
        recording_dirs = sorted([d for d in potato_dir.iterdir() if d.is_dir() and d.name.startswith('rec_')])

        if not recording_dirs:
            print(f"  Warning: No 'rec_...' subdirectories found in {potato_dir}")
            continue

        print(f"  Found {len(recording_dirs)} recording directories.")

        for rec_dir in recording_dirs:
            match = re.match(r'^rec_(\d{8}_\d{6})$', rec_dir.name)
            if not match:
                print(f"    Warning: Skipping directory with unexpected name format: {rec_dir.name}")
                continue

            timestamp = match.group(1) # Extract YYYYMMDD_HHMMSS
            print(f"    Processing recording: {timestamp}")

            # --- Define source file paths ---
            csi_file_name = f"csi_data_{timestamp}.csv"
            src_csi_path = rec_dir / csi_file_name
            src_img_path = rec_dir / "capture.png"
            gt_file_name = f"{timestamp}.npy"
            src_gt_path = gt_dir_path / gt_file_name

            # --- Check if all source files exist ---
            files_missing = False
            if not src_csi_path.is_file():
                print(f"      Error: CSI file missing: {src_csi_path}")
                files_missing = True
            if not src_img_path.is_file():
                print(f"      Warning: Capture image missing: {src_img_path}")
                # Decide if you want to skip if image is missing, or just warn
                # files_missing = True # Uncomment to skip if image is mandatory
            if not src_gt_path.is_file():
                print(f"      Error: Ground truth file missing: {src_gt_path}")
                files_missing = True

            if files_missing:
                print(f"      Skipping recording {timestamp} due to missing files.")
                total_recordings_skipped += 1
                continue

            # --- Define destination paths ---
            output_rec_path = output_dataset_path / timestamp # Subfolder named just with timestamp
            output_rec_path.mkdir(parents=True, exist_ok=True)

            dest_csi_path = output_rec_path / "csi_data.csv" # Consistent name
            dest_img_path = output_rec_path / "capture.png"  # Keep original name
            dest_gt_path = output_rec_path / "ground_truth.npy" # Consistent name

            # --- Copy files ---
            try:
                print(f"      Copying CSI: {src_csi_path} -> {dest_csi_path}")
                shutil.copy2(src_csi_path, dest_csi_path) # copy2 preserves metadata

                print(f"      Copying IMG: {src_img_path} -> {dest_img_path}")
                shutil.copy2(src_img_path, dest_img_path)

                print(f"      Copying GT:  {src_gt_path} -> {dest_gt_path}")
                shutil.copy2(src_gt_path, dest_gt_path)

                total_recordings_processed += 1
            except Exception as e:
                print(f"      Error copying files for recording {timestamp}: {e}")
                traceback.print_exc()
                total_recordings_skipped += 1

    print("\n------------------------------------")
    print("Data organization complete.")
    print(f"Total recordings processed successfully: {total_recordings_processed}")
    print(f"Total recordings skipped (missing files): {total_recordings_skipped}")
    print("------------------------------------")


if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = Path(__file__).parent.resolve()
    # Assume the script is in 'scripts' and 'raw_data'/'processed_data' are siblings
    project_root = script_dir.parent

    # Define default paths relative to the project root
    default_raw_data = project_root / "raw_data"
    default_processed_data = project_root / "processed_data"
    print("test")
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Organize raw CSI, GT, and image data into a processed structure.")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=str(default_raw_data),
        help=f"Path to the main raw_data directory (default: {default_raw_data})"
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=str(default_processed_data),
        help=f"Path to the main processed_data directory (default: {default_processed_data})"
    )

    args = parser.parse_args()

    organize_data(args.raw_dir, args.processed_dir)