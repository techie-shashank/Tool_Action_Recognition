import matplotlib.pyplot as plt
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
                if 'classes' in rec_data:
                    label_names = {k: v for k, v in rec_data['classes'].items()}
                    print("Label mapping:")
                    for label, count in label_counts.items():
                        name = label_names.get(label, f"Unknown({label})")
                        print(f"  {name}: {count} samples")

            # Visualization (only for requested sensor)
            if sensor_type == sensor:
                plt.figure(figsize=(12, 8))

                # Plot all sensor channels
                for i, col in enumerate(sensor_df.columns):
                    if col in ['time [s]', 'label']:
                        continue

                    plt.plot(sensor_df['time [s]'], sensor_df[col], label=col)

                # Add label regions if available
                if 'label' in sensor_df.columns:
                    unique_labels = sensor_df['label'].unique()
                    current_label = None
                    start_time = None

                    for idx, row in sensor_df.iterrows():
                        if current_label != row['label']:
                            if current_label is not None:
                                # Shade previous region
                                plt.axvspan(start_time, row['time [s]'],
                                            alpha=0.1, color=plt.cm.tab10(current_label % 10))
                                plt.text((start_time + row['time [s]']) / 2,
                                         plt.ylim()[1] * 0.95,
                                         label_names.get(current_label, current_label),
                                         ha='center')

                            current_label = row['label']
                            start_time = row['time [s]']

                plt.title(f"Recording {rec_id} - {sensor_type} Sensor")
                plt.xlabel("Time [s]")
                plt.ylabel("Value")
                plt.legend()
                plt.tight_layout()
                plt.show()


if __name__ == "__main__":
    explore_data(tool='electric_screwdriver', sensor='acc')