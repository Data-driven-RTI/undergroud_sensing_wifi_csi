import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import ast  # For safely parsing the string list

# --- Configuration ---
# Define column names based on your header
COLUMN_NAMES = [
    'timestamp_script', 'receiver_node', 'sender_node', 'channel_index', 'rssi',
    'rate', 'sig_mode', 'mcs', 'bandwidth', 'noise_floor',
    'node_timestamp_us', 'csi_data_array'
]

# --- Helper Functions ---

def parse_csi_string(csi_string: str) -> np.ndarray | None:
    """
    Parses the CSI data string "[I1, Q1, I2, Q2, ...]" into a numpy array
    of complex numbers [I1 + 1j*Q1, I2 + 1j*Q2, ...].
    """
    try:
        # Safely evaluate the string literal into a Python list
        data_list = ast.literal_eval(csi_string)

        # Convert to numpy array of floats
        data_array = np.array(data_list, dtype=float)

        # Check if the length is even for I/Q pairing
        if len(data_array) % 2 != 0:
            print(f"Warning: CSI data length is not even ({len(data_array)}). Skipping pairing for this entry.", file=sys.stderr)
            # Return raw data or handle as needed - here returning None
            return None

        # Reshape into pairs (N, 2) where N is the number of subcarriers
        iq_pairs = data_array.reshape(-1, 2)

        # Combine into complex numbers: I + jQ
        complex_csi = iq_pairs[:, 0] + 1j * iq_pairs[:, 1]
        return complex_csi

    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Error parsing CSI string: {csi_string[:50]}... Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Unexpected error parsing CSI string: {csi_string[:50]}... Error: {e}", file=sys.stderr)
        return None


def calculate_amplitude_phase(complex_csi: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Calculates amplitude and phase from complex CSI data."""
    if complex_csi is None:
        return None, None
    amplitude = np.abs(complex_csi)
    phase = np.angle(complex_csi) # Radians
    # phase = np.unwrap(phase) # Optional: Unwrap phase to avoid jumps
    return amplitude, phase


def load_csi_data(filepath: str) -> pd.DataFrame | None:
    """Loads and preprocesses the CSI data from a CSV file."""
    print(f"Loading data from: {filepath}")
    try:
        # Load assuming the file DOES NOT contain the header row, use our names
        # If your file DOES contain the header, use: header=0 instead of header=None
        df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES)
        print(f"Successfully loaded {len(df)} rows.")

        # --- Data Type Conversions and Cleaning ---
        # Convert timestamp_script to datetime objects
        try:
            df['timestamp_script'] = pd.to_datetime(df['timestamp_script'])
            df = df.sort_values(by='timestamp_script').reset_index(drop=True)
        except Exception as e:
            print(f"Warning: Could not parse 'timestamp_script'. Error: {e}", file=sys.stderr)

        # Convert relevant columns to numeric, coercing errors to NaN
        numeric_cols = ['receiver_node', 'sender_node', 'channel_index', 'rssi', 'rate',
                        'sig_mode', 'mcs', 'bandwidth', 'noise_floor', 'node_timestamp_us']
        for col in numeric_cols:
            if col in df.columns:
                 # Use 'integer' for known ints, 'float' otherwise if NaNs might appear
                dtype_target = 'integer' if col not in ['rssi', 'noise_floor'] else 'float'
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype_target)
                except Exception as e:
                     print(f"Warning: Could not convert column '{col}' to numeric. Error: {e}. Trying float.", file=sys.stderr)
                     try:
                         df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                     except Exception as e_float:
                         print(f"Error: Failed converting '{col}' to float as well. Error: {e_float}", file=sys.stderr)


        # --- Parse CSI Data ---
        print("Parsing CSI data strings...")
        df['csi_complex'] = df['csi_data_array'].apply(parse_csi_string)

        # Remove rows where CSI parsing failed
        original_len = len(df)
        df = df.dropna(subset=['csi_complex']).reset_index(drop=True)
        if len(df) < original_len:
            print(f"Removed {original_len - len(df)} rows due to CSI parsing errors.")

        # Calculate Amplitude and Phase
        print("Calculating CSI amplitude and phase...")
        # Note: This creates lists of arrays in the columns
        amp_phase = df['csi_complex'].apply(calculate_amplitude_phase)
        df['csi_amplitude'] = amp_phase.apply(lambda x: x[0])
        df['csi_phase'] = amp_phase.apply(lambda x: x[1])

        # Drop intermediate complex column if desired to save memory
        # df = df.drop(columns=['csi_complex'])

        print("Preprocessing complete.")
        return df

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}", file=sys.stderr)
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty: {filepath}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}", file=sys.stderr)
        return None

# --- Analysis and Plotting Functions ---

def plot_rssi_over_time(df: pd.DataFrame):
    """Plots RSSI values against the script timestamp."""
    if 'timestamp_script' not in df.columns or 'rssi' not in df.columns:
        print("Error: Missing 'timestamp_script' or 'rssi' columns for plotting.")
        return

    plt.figure(figsize=(12, 5))
    plt.plot(df['timestamp_script'], df['rssi'], marker='.', linestyle='-', markersize=3)
    plt.title('RSSI over Time')
    plt.xlabel('Timestamp (Script)')
    plt.ylabel('RSSI (dBm)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show() # Show plot immediately

def plot_csi_amplitude_example(df: pd.DataFrame, num_examples: int = 5):
    """Plots the CSI amplitude for a few example packets."""
    if 'csi_amplitude' not in df.columns or df['csi_amplitude'].isnull().all():
        print("Error: Missing or empty 'csi_amplitude' data for plotting.")
        return

    plot_indices = np.linspace(0, len(df) - 1, num_examples, dtype=int)

    plt.figure(figsize=(12, 6))
    for i in plot_indices:
        amplitude_data = df.loc[i, 'csi_amplitude']
        if amplitude_data is not None and len(amplitude_data) > 0:
            subcarrier_indices = np.arange(len(amplitude_data))
            timestamp_label = df.loc[i, 'timestamp_script'].strftime('%H:%M:%S.%f')[:-3] if 'timestamp_script' in df.columns else f"Index {i}"
            plt.plot(subcarrier_indices, amplitude_data, marker='.', linestyle='-', markersize=4, label=f'Packet at {timestamp_label}')
        else:
             print(f"Warning: No valid amplitude data at index {i}")

    plt.title(f'CSI Amplitude vs. Subcarrier Index ({num_examples} Examples)')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Amplitude (Linear)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show() # Show plot immediately

def plot_csi_heatmap(df: pd.DataFrame, data_type: str = 'amplitude', max_packets: int = 500):
    """Creates a heatmap of CSI amplitude or phase over time."""
    col_name = f'csi_{data_type}'
    if col_name not in df.columns or df[col_name].isnull().all():
        print(f"Error: Missing or empty '{col_name}' data for heatmap.")
        return

    # Limit packets for performance/visualization
    df_subset = df.head(max_packets)
    csi_matrix = df_subset[col_name].tolist()

    # Check if all lists have the same length, pad or truncate if necessary (or error out)
    try:
        csi_matrix = np.array(csi_matrix)
    except ValueError:
        print("Error creating heatmap: CSI arrays have inconsistent lengths.", file=sys.stderr)
        # Attempt to find the most common length and pad/truncate? (More complex)
        # For now, we just fail.
        return

    if csi_matrix.ndim != 2:
        print(f"Error: Could not form a 2D matrix from '{col_name}'. Shape: {csi_matrix.shape}", file=sys.stderr)
        return

    plt.figure(figsize=(12, 7))
    plt.imshow(csi_matrix.T, aspect='auto', cmap='viridis', origin='lower',
               interpolation='nearest') # Transpose to have time on x-axis, subcarriers on y
    plt.colorbar(label=f'{data_type.capitalize()} ({ "Linear" if data_type=="amplitude" else "Radians" })')
    plt.title(f'CSI {data_type.capitalize()} Heatmap (First {len(df_subset)} packets)')
    plt.xlabel('Packet Index (Time)')
    plt.ylabel('Subcarrier Index')
    plt.tight_layout()
    # plt.show() # Show plot immediately


def print_summary(df: pd.DataFrame):
    """Prints a summary of the loaded data."""
    print("\n" + "="*20 + " Data Summary " + "="*20)
    print(f"Total packets loaded and parsed: {len(df)}")

    if not df.empty:
        print("\nBasic Info:")
        df.info(memory_usage='deep') # Shows types and non-null counts

        print("\nNumeric Columns Statistics:")
        # Select only columns that are actually numeric AFTER potential coercion
        numeric_cols_present = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols_present:
             print(df[numeric_cols_present].describe())
        else:
            print("No purely numeric columns found after loading/conversion.")


        print("\nValue Counts for Key Categorical/Index Columns:")
        categorical_cols = ['receiver_node', 'sender_node', 'channel_index', 'sig_mode', 'mcs', 'bandwidth']
        for col in categorical_cols:
            if col in df.columns and not df[col].isnull().all():
                 print(f"\n--- {col} ---")
                 # Limit displayed values if there are too many unique ones
                 counts = df[col].value_counts()
                 print(counts.head(10))
                 if len(counts) > 10:
                     print(f"... and {len(counts)-10} more unique values.")
            # else:
            #     print(f"Column '{col}' not found or is all null.")

        # Example CSI dimensions
        if 'csi_amplitude' in df.columns and not df['csi_amplitude'].isnull().all():
            first_valid_amp = df['csi_amplitude'].dropna().iloc[0]
            if first_valid_amp is not None:
                print(f"\nNumber of subcarriers reported (example): {len(first_valid_amp)}")

    else:
        print("DataFrame is empty, no summary available.")
    print("="*54 + "\n")


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Analyze CSI data log files (CSV format).")
    parser.add_argument("filepath", help="Path to the CSI data log file (.csv)")
    parser.add_argument("-p", "--plot", action="store_true", help="Generate and show plots.")
    parser.add_argument("--max_heatmap_packets", type=int, default=500,
                        help="Maximum number of packets to include in the heatmap plot for performance.")

    args = parser.parse_args()

    csi_df = load_csi_data(args.filepath)

    if csi_df is not None and not csi_df.empty:
        print_summary(csi_df)

        if args.plot:
            print("Generating plots...")
            plot_rssi_over_time(csi_df)
            plot_csi_amplitude_example(csi_df, num_examples=5)
            plot_csi_heatmap(csi_df, data_type='amplitude', max_packets=args.max_heatmap_packets)
            # plot_csi_heatmap(csi_df, data_type='phase') # Optional phase heatmap
            print("Displaying plots...")
            plt.show() # Display all generated figures
        else:
            print("\nRun with -p or --plot to generate visualizations.")

    else:
        print("Exiting due to loading errors or empty data.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()