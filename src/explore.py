import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data.loader import ToolTrackingDataLoader


def explore_data(tool, sensor):
    # Load data without segmentation
    loader = ToolTrackingDataLoader(source=r"C:\Users\taimo\Desktop\FAU-Courses\ADLTS-Seminar\data\tool-tracking-data")
    data_dict = loader.load_measurement_data(tool)

    print(f"Exploring tool: {tool}")
    print(f"Available recordings: {list(data_dict.keys())}")

    # Analyze each recording
    for rec_id, rec_data in data_dict.items():
        if rec_id == 'classes':  # Skip class definitions
            continue

        print(f"\nRecording {rec_id}:")

        # Sensor types in this recording
        sensor_types = [k for k in rec_data.keys() if k != 'classes']
        print(f"Sensor types: {sensor_types}")

        # Analyze each sensor type
        for sensor_type in sensor_types:
            sensor_df = rec_data[sensor_type]
            print(f"\nSensor: {sensor_type}")
            print(f"Shape: {sensor_df.shape}")

            # Basic statistics
            print("Column statistics:")
            for col in sensor_df.columns:
                if col == 'label' or col == 'time [s]':
                    continue

                values = sensor_df[col]
                print(f"  {col}: min={values.min():.4f}, max={values.max():.4f}, "
                      f"mean={values.mean():.4f}, std={values.std():.4f}, "
                      f"nan={values.isna().sum()}/{len(values)}")

            # Label distribution
            if 'label' in sensor_df.columns:
                label_counts = sensor_df['label'].value_counts()
                print("\nLabel distribution:")
                print(label_counts)

                # Map labels to names if available
                label_names = {}
                if 'classes' in rec_data:
                    label_names = {k: v for k, v in rec_data['classes'].items()}
                    print("Label mapping:")
                    for label, count in label_counts.items():
                        name = label_names.get(label, f"Unknown({label})")
                        print(f"  {name}: {count} samples")

                # Plot label distribution as bar plot
                plt.figure(figsize=(max(8, len(label_counts) * 1.5), 5))
                labels = [label_names.get(label, str(label)) for label in label_counts.index]
                counts = label_counts.values

                bars = plt.bar(labels, counts, color='skyblue', edgecolor='black')
                plt.title(f"Label Distribution for Recording {rec_id} - {sensor_type}")
                plt.xlabel("Label")
                plt.ylabel("Count")
                plt.xticks(range(len(labels)),[str(i) for i in range(len(labels))])

                # Write label names on top of each bar
                for i, (bar, name) in enumerate(zip(bars, labels)):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5, name, ha='center', va='bottom',
                             rotation=0, fontsize=8)

                plt.tight_layout()
                plt.show()

            if sensor_type == sensor:
                try:
                    t = sensor_df["time [s]"]
                    x_mea = sensor_df.drop(columns=["time [s]", "label"], errors="ignore")
                    plt.figure(figsize=(24, 4))
                    plt.title(f"Measurement {rec_id} - {sensor_type}")
                    plt.plot(t, x_mea)
                    plt.xlabel('Time [s]')
                    plt.ylabel(f'Sensor values ({sensor_type})')
                    plt.legend(x_mea.columns, loc="upper right")
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Failed to plot measurement for {sensor_type} in recording {rec_id}: {e}")


def create_sliding_windows(ts_data, window_size, step_size):
    """
    ts_data: np.ndarray of shape (N, D) with time in first column
    window_size: number of time steps per window
    step_size: how much to shift the window
    returns: list of windows (each shape (window_size, D))
    """
    windows = []
    for start in range(0, len(ts_data) - window_size + 1, step_size):
        window = ts_data[start:start + window_size]
        windows.append(window)
    return np.array(windows)


def explore_data_one_window(tool, sensor, window_size=100, step_size=50):
    # Load data
    loader = ToolTrackingDataLoader(source=r"C:\Users\taimo\Desktop\FAU-Courses\ADLTS-Seminar\data\tool-tracking-data")
    data_dict = loader.load_measurement_data(tool)

    print(f"Exploring tool: {tool}")
    print(f"Available recordings: {list(data_dict.keys())}")

    # Pick one measurement for segmentation
    sample_rec_id = "01"
    if sample_rec_id not in data_dict:
        print("Recording '01' not found. Cannot generate sample window.")
        return

    rec_data = data_dict[sample_rec_id]
    if sensor not in rec_data:
        print(f"Sensor '{sensor}' not found in recording {sample_rec_id}")
        return

    # Extract raw measurement for this sensor
    df = rec_data[sensor]
    ts_array = df.drop(columns="label", errors="ignore").to_numpy()

    # Segment into sliding windows
    Xt_acc = create_sliding_windows(ts_array, window_size=window_size, step_size=step_size)
    print(f"Segmented into {len(Xt_acc)} windows with shape {Xt_acc[0].shape}")

    # Plot full measurement and one sample window
    t_mea = ts_array[:, 0]
    x_mea = ts_array[:, 1:]

    fig, axs = plt.subplots(1, 2, figsize=(24, 4))

    # Full measurement
    axs[0].set_title(f"Full Measurement {sample_rec_id} - {sensor}")
    axs[0].plot(t_mea, x_mea)
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel(f"Sensor values ({sensor})")
    axs[0].legend(df.columns[1:], loc="upper right")

    # One window
    t_win = Xt_acc[70][:, 0] if len(Xt_acc) > 70 else Xt_acc[0][:, 0]
    x_win = Xt_acc[70][:, 1:] if len(Xt_acc) > 70 else Xt_acc[0][:, 1:]
    axs[1].set_title("Single Window Sample")
    axs[1].plot(t_win, x_win)
    axs[1].set_xlabel("Time [a.u.]")
    axs[1].set_ylabel(f"Sensor values ({sensor})")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    explore_data(tool='electric_screwdriver', sensor='acc')

# # Visualization (only for requested sensor)
            # if sensor_type == sensor:
            #     plt.figure(figsize=(12, 8))
            #
            #     # Plot all sensor channels
            #     for i, col in enumerate(sensor_df.columns):
            #         if col in ['time [s]', 'label']:
            #             continue
            #
            #         plt.plot(sensor_df['time [s]'], sensor_df[col], label=col)
            #
            #     # Add label regions if available
            #     if 'label' in sensor_df.columns:
            #         unique_labels = sensor_df['label'].unique()
            #         current_label = None
            #         start_time = None
            #
            #         for idx, row in sensor_df.iterrows():
            #             if current_label != row['label']:
            #                 if current_label is not None:
            #                     # Shade previous region
            #                     plt.axvspan(start_time, row['time [s]'],
            #                                 alpha=0.1, color=plt.cm.tab10(current_label % 10))
            #                     plt.text((start_time + row['time [s]']) / 2,
            #                              plt.ylim()[1] * 0.95,
            #                              label_names.get(current_label, current_label),
            #                              ha='center')
            #
            #                 current_label = row['label']
            #                 start_time = row['time [s]']
            #
            #     plt.title(f"Recording {rec_id} - {sensor_type} Sensor")
            #     plt.xlabel("Time [s]")
            #     plt.ylabel("Value")
            #     plt.legend()
            #     plt.tight_layout()
            #     plt.show()