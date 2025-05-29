# scripts/plot_rssi_over_time.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import traceback

def plot_rssi_over_packets(
    csv_file_path: str | Path,
    sender_node: int,
    receiver_node: int,
    channel_index: int,
    output_path: str | Path | None = None,
    title: str | None = None,
    max_packets: int | None = None # Optional limit on packets to plot
) -> bool:
    """
    Loads raw data from a CSV, filters for a specific link and channel,
    and plots the RSSI values over the packets (time).

    Args:
        csv_file_path: Path to the raw 'csi_data_*.csv' file (contains RSSI).
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
        # Read only necessary columns for RSSI plotting
        columns_to_read = ['sender_node', 'receiver_node', 'channel_index', 'rssi']
        df = pd.read_csv(csv_file, usecols=columns_to_read, low_memory=False)
        print(f"Read {len(df)} rows.")

        # Convert types for filtering (handle potential errors)
        for col in ['sender_node', 'receiver_node', 'channel_index', 'rssi']: # Include rssi here
             df[col] = pd.to_numeric(df[col], errors='coerce')
        # Drop rows where any essential column is NaN (including rssi if it was non-numeric)
        df.dropna(subset=columns_to_read, inplace=True)
        # Convert relevant columns to int after cleaning
        df = df.astype({'sender_node': int, 'receiver_node': int, 'channel_index': int, 'rssi': int})


        # Filter for the specific link and channel
        df_filtered = df[
            (df['sender_node'] == sender_node) &
            (df['receiver_node'] == receiver_node) &
            (df['channel_index'] == channel_index)
        ].copy() # Use .copy()

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


        # Extract RSSI values for the filtered packets
        rssi_values = df_filtered['rssi'].to_numpy()
        print(f"Extracted {len(rssi_values)} RSSI values.")

        if len(rssi_values) == 0: # Should be caught earlier, but double-check
             print("Error: No valid RSSI values found after filtering.")
             return False

        # --- Plotting ---
        fig = plt.figure(figsize=(15, 7)) # Assign the figure object
        packet_indices = np.arange(len(rssi_values))

        plt.plot(
            packet_indices,
            rssi_values,
            marker='.', # Show individual points
            linestyle='-',
            color='dodgerblue', # Choose a color for RSSI
            label=f'RSSI Ch {channel_index}'
        )

        # Set plot title
        csv_name_part = csv_file.stem # e.g., csi_data_20250410_113951
        if title is None:
            plot_title = (f'RSSI over Packets\n'
                          f'Link: {sender_node} -> {receiver_node}, Channel: {channel_index} ({len(rssi_values)} packets)\n'
                          f'File: {csv_name_part}')
        else:
            plot_title = title

        plt.title(plot_title)
        plt.xlabel("Packet Index")
        plt.ylabel("RSSI (dBm)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # --- Determine Save Path ---
        if output_path:
            save_path = Path(output_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Default: Save in the script's directory
            default_filename = f"rssi_vs_packets_{csv_name_part}_link_{sender_node}_{receiver_node}_ch{channel_index}.png"
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
         # Check if the error is specifically about 'rssi' column
         try:
             cols_found = pd.read_csv(csv_file, nrows=1).columns.tolist()
             print(f"Error: Missing expected column in CSV: {e}. Columns found: {cols_found}")
         except Exception: # Handle errors reading columns if file is problematic
             print(f"Error: Missing expected column in CSV: {e}. Could not read columns.")
         return False
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        if fig: # If the figure was created before the error, close it
             plt.close(fig)
        return False


# --- Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RSSI values over packets for a specific link and channel from a raw CSV file.")
    # Arguments are the same as the CSI script
    parser.add_argument("csv_file", type=str, help="Path to the raw csi_data_*.csv file.")
    parser.add_argument("sender", type=int, help="Sender node index (0-15).")
    parser.add_argument("receiver", type=int, help="Receiver node index (0-15).")
    parser.add_argument("channel", type=int, help="Channel index (e.g., 1, 6, 11).")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Optional path to save the plot image. If omitted, saves to the script's directory.")
    parser.add_argument("--title", "-t", type=str, default=None, help="Optional custom title for the plot.")
    parser.add_argument("--max_packets", "-n", type=int, default=None, help="Optional: Maximum number of packets to plot.")


    args = parser.parse_args()

    print(f"Running plot script for RSSI over packets...")

    success = plot_rssi_over_packets(
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