# scripts/plot_subcarriers_over_time.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import traceback
import re # For parsing CSI string

# --- Configuration ---

# !! IMPORTANT: Copy the exact definition from your process_all_data.py !!
# !! Indices for the 52 commonly used DATA subcarriers in HT20           !!
_HT20_PILOT_OFFSETS = {-21, -7, 7, 21}
_HT20_NULL_DC_OFFSET = {0}
_all_potential_data_pilot_offsets = set(range(-28, 29)) # -28 to +28 inclusive
_data_relative_offsets = _all_potential_data_pilot_offsets - _HT20_NULL_DC_OFFSET - _HT20_PILOT_OFFSETS
_center_index_in_64 = 32 # Array index corresponding to subcarrier 0 (DC)
DATA_SUBCARRIER_INDICES = sorted(
    [_center_index_in_64 + offset for offset in _data_relative_offsets]
)
NUM_CSI_SUBCARRIERS = len(DATA_SUBCARRIER_INDICES)

if NUM_CSI_SUBCARRIERS != 52:
     print(f"ERROR: DATA_SUBCARRIER_INDICES length is {NUM_CSI_SUBCARRIERS}, expected 52. Check definition.")
     exit(1)
# --------------------------------------------------------------------

# --- Helper Function: Parse CSI String (Keep as is) ---
def parse_csi_array_string(csi_str):
    """Parses the '[i,q,i,q,...]' string into a numpy array of complex numbers."""
    if not isinstance(csi_str, str) or not csi_str.startswith('[') or not csi_str.endswith(']'): return None
    csi_str = csi_str[1:-1]
    try:
        values = np.fromstring(csi_str, dtype=int, sep=',')
        if values.size != 128: return None # Expect HT20 size (64*2)
        complex_csi = values.astype(np.float32).view(np.complex64)
        # Ensure we have enough values for the maximum index needed
        if len(complex_csi) <= max(DATA_SUBCARRIER_INDICES):
             # print(f"Warning: Parsed complex CSI length {len(complex_csi)} is too short for max index {max(DATA_SUBCARRIER_INDICES)}. Skipping.")
             return None
        return complex_csi
    except ValueError: return None
    except Exception as e: print(f"Error parsing CSI string: {e}"); traceback.print_exc(); return None


def plot_subcarriers_over_packets(
    csv_file_path: str | Path,
    sender_node: int,
    receiver_node: int,
    channel_index: int,
    output_path: str | Path | None = None,
    title: str | None = None,
    max_packets: int | None = None # Optional limit on packets to plot
) -> bool:
    """
    Loads raw CSI data from a CSV, filters for a specific link and channel,
    and plots the amplitude of each subcarrier over the packets (time).

    Args:
        csv_file_path: Path to the raw 'csi_data_*.csv' file.
        sender_node: Sender node index (0-15).
        receiver_node: Receiver node index (0-15).
        channel_index: Channel index (e.g., 1, 6, or 11).
        output_path: Optional path to save the plot image. If None, saves to a
                     default location within the script's directory.
        title: Optional custom title for the plot.
        max_packets: Optional maximum number of packets to process and plot.

    Returns:
        True if plotting and saving were successful, False otherwise.
    """
    csv_file = Path(csv_file_path)
    if not csv_file.is_file():
        print(f"Error: Raw CSV file not found: {csv_file}")
        return False

    # --- Determine script directory for default save location ---
    try:
        script_dir = Path(__file__).parent.resolve()
    except NameError:
        script_dir = Path('.').resolve()
        print(f"Warning: Could not determine script directory via __file__. Defaulting save location to current directory: {script_dir}")

    fig = None
    try:
        print(f"Reading raw CSV: {csv_file}...")
        # Read only necessary columns
        columns_to_read = ['sender_node', 'receiver_node', 'channel_index', 'csi_data_array']
        df = pd.read_csv(csv_file, usecols=columns_to_read, low_memory=False)
        print(f"Read {len(df)} rows.")

        # Convert types for filtering (handle potential errors)
        for col in ['sender_node', 'receiver_node', 'channel_index']:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['sender_node', 'receiver_node', 'channel_index'], inplace=True)
        df = df.astype({'sender_node': int, 'receiver_node': int, 'channel_index': int})


        # Filter for the specific link and channel
        df_filtered = df[
            (df['sender_node'] == sender_node) &
            (df['receiver_node'] == receiver_node) &
            (df['channel_index'] == channel_index)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning if modified later

        num_found_packets = len(df_filtered)
        print(f"Found {num_found_packets} packets for link {sender_node}->{receiver_node} on channel {channel_index}.")

        if num_found_packets == 0:
            print("No matching packets found. Cannot generate plot.")
            return False

        # Apply max_packets limit if specified
        if max_packets is not None and num_found_packets > max_packets:
            print(f"Limiting plot to the first {max_packets} packets.")
            df_filtered = df_filtered.head(max_packets)
            num_packets_to_plot = max_packets
        else:
            num_packets_to_plot = num_found_packets


        # Extract CSI amplitudes for each packet
        all_packet_amplitudes = []
        packets_processed = 0
        packets_skipped = 0
        required_csi_len = max(DATA_SUBCARRIER_INDICES) + 1 if DATA_SUBCARRIER_INDICES else 0

        print(f"Processing {num_packets_to_plot} packets...")
        for index, row in df_filtered.iterrows():
            complex_csi = parse_csi_array_string(row['csi_data_array'])
            if complex_csi is None:
                packets_skipped += 1
                continue
            # This check might already be handled in parse_csi_array_string, but double-check
            if len(complex_csi) < required_csi_len:
                 packets_skipped += 1
                 continue

            try:
                # Select the 52 data subcarriers
                selected_complex = complex_csi[DATA_SUBCARRIER_INDICES]
                # Calculate amplitudes
                amplitudes = np.abs(selected_complex).astype(np.float32) # Shape (52,)

                if amplitudes.shape == (NUM_CSI_SUBCARRIERS,):
                    all_packet_amplitudes.append(amplitudes)
                    packets_processed += 1
                else:
                    print(f"Warning: Unexpected amplitude shape {amplitudes.shape} for row {index}. Skipping.")
                    packets_skipped += 1

            except IndexError:
                 print(f"Warning: IndexError accessing subcarriers for row {index}. CSI length: {len(complex_csi)}. Skipping.")
                 packets_skipped += 1
            except Exception as e:
                 print(f"Warning: Error processing row {index}: {e}. Skipping.")
                 packets_skipped += 1


        if packets_skipped > 0:
             print(f"Note: Skipped {packets_skipped} packets during CSI processing.")

        if not all_packet_amplitudes:
            print("Error: No valid CSI data could be extracted after processing. Cannot plot.")
            return False

        # Stack into a 2D array: (num_valid_packets, num_subcarriers)
        amplitude_matrix = np.stack(all_packet_amplitudes, axis=0)
        num_valid_packets = amplitude_matrix.shape[0]
        print(f"Created amplitude matrix with shape: {amplitude_matrix.shape}")

        # --- Plotting ---
        fig = plt.figure(figsize=(15, 7)) # Assign the figure object
        packet_indices = np.arange(num_valid_packets)

        # Use a colormap to distinguish subcarriers (will be busy with 52!)
        colors = plt.cm.viridis(np.linspace(0, 1, NUM_CSI_SUBCARRIERS))

        for i in range(NUM_CSI_SUBCARRIERS):
            plt.plot(
                packet_indices,
                amplitude_matrix[:, i], # Plot amplitude of subcarrier 'i' over packets
                color=colors[i],
                # label=f'Subcarrier {i}', # Legend would be too crowded
                alpha=0.7 # Add some transparency
            )

        # Set plot title
        csv_name_part = csv_file.stem # e.g., csi_data_20250410_113951
        if title is None:
            plot_title = (f'CSI Subcarrier Amplitudes over Packets\n'
                          f'Link: {sender_node} -> {receiver_node}, Channel: {channel_index} ({num_valid_packets} packets)\n'
                          f'File: {csv_name_part}')
        else:
            plot_title = title

        plt.title(plot_title)
        plt.xlabel("Packet Index")
        plt.ylabel("CSI Amplitude")
        # plt.legend() # Avoid legend for 52 lines
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # --- Determine Save Path ---
        if output_path:
            save_path = Path(output_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Default: Save in the script's directory
            default_filename = f"subcarriers_vs_packets_{csv_name_part}_link_{sender_node}_{receiver_node}_ch{channel_index}.png"
            save_path = script_dir / default_filename
            print(f"No output path specified. Saving to default location: {save_path}")

        # --- Save the plot ---
        plt.savefig(save_path)
        print(f"Plot saved successfully to: {save_path}")
        plt.close(fig) # Close the specific figure
        return True

    except FileNotFoundError:
         print(f"Error: Raw CSV file not found at the specified path: {csv_file}")
         return False
    except pd.errors.EmptyDataError:
         print(f"Error: Raw CSV file is empty: {csv_file}")
         return False
    except KeyError as e:
         print(f"Error: Missing expected column in CSV: {e}. Columns found: {df.columns.tolist()}")
         return False
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        if fig: # If the figure was created before the error, close it
             plt.close(fig)
        return False


# --- Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot CSI subcarrier amplitudes over packets for a specific link and channel from a raw CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the raw csi_data_*.csv file.")
    parser.add_argument("sender", type=int, help="Sender node index (0-15).")
    parser.add_argument("receiver", type=int, help="Receiver node index (0-15).")
    parser.add_argument("channel", type=int, help="Channel index (e.g., 1, 6, 11).")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Optional path to save the plot image. If omitted, saves to the script's directory.")
    parser.add_argument("--title", "-t", type=str, default=None, help="Optional custom title for the plot.")
    parser.add_argument("--max_packets", "-n", type=int, default=None, help="Optional: Maximum number of packets to plot.")


    args = parser.parse_args()

    print(f"Running plot script for subcarriers over packets...")

    success = plot_subcarriers_over_packets(
        csv_file_path=args.csv_file,
        sender_node=args.sender,
        receiver_node=args.receiver,
        channel_index=args.channel,
        output_path=args.output,
        title=args.title,
        max_packets=args.max_packets
    )

    if success:
        print("Plotting completed successfully.")
        exit(0)
    else:
        print("Plotting failed.")
        exit(1)