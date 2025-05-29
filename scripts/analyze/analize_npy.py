import numpy as np
import argparse
from pathlib import Path
import sys

def check_npy_file(file_path: Path):
    """
    Loads a .npy file and returns its dtype and shape.

    Args:
        file_path: Path object pointing to the .npy file.

    Returns:
        A tuple (dtype, shape) if successful, otherwise None.
        Prints an error message if loading fails.
    """
    try:
        # Load the array without loading the full data into memory immediately
        # if possible (though for dtype/shape, it usually needs to read metadata)
        # Use allow_pickle=False for security unless you trust the source.
        # For just checking metadata, allow_pickle=True might be needed
        # if the dtype is object, but be cautious. Let's try False first.
        try:
            data = np.load(file_path, allow_pickle=False)
        except ValueError as e:
            # If allow_pickle=False fails and error mentions pickling, try True
            if "allow_pickle=True" in str(e):
                print(f"Warning: Retrying {file_path.name} with allow_pickle=True...", file=sys.stderr)
                data = np.load(file_path, allow_pickle=True)
            else:
                raise # Re-raise other ValueErrors

        dtype = data.dtype
        shape = data.shape
        return dtype, shape
    except Exception as e:
        print(f"Error loading or processing file '{file_path}': {e}", file=sys.stderr)
        return None, None

def main():
    parser = argparse.ArgumentParser(
        description="Check the data type (dtype) and shape (including length) of .npy files in a directory."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The directory containing .npy files to check."
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search recursively through subdirectories."
    )

    args = parser.parse_args()

    search_dir = Path(args.directory)

    if not search_dir.is_dir():
        print(f"Error: Directory not found: {search_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {'recursively ' if args.recursive else ''}in directory: {search_dir}")
    print("-" * 60)

    # Choose the glob pattern based on the recursive flag
    if args.recursive:
        npy_files = search_dir.rglob('*.npy')
    else:
        npy_files = search_dir.glob('*.npy')

    found_count = 0
    error_count = 0

    for npy_file in npy_files:
        if npy_file.is_file(): # Ensure it's actually a file
            found_count += 1
            print(f"Checking: {npy_file}")
            dtype, shape = check_npy_file(npy_file)
            if dtype is not None and shape is not None:
                # Determine 'length' - usually the size of the first dimension
                length_str = "N/A (0-dim)" if len(shape) == 0 else str(shape[0])
                print(f"  -> Dtype: {dtype}")
                print(f"  -> Shape: {shape}")
                print(f"  -> Length (1st dim): {length_str}")
                print("-" * 20)
            else:
                error_count += 1
                print("-" * 20)


    print("-" * 60)
    print(f"Scan complete. Found {found_count} potential .npy files.")
    if error_count > 0:
        print(f"Encountered errors while processing {error_count} files.")

if __name__ == "__main__":
    main()