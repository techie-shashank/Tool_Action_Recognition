import numpy as np
from datatools import (
    Tool, Config, MeasurementSeries, MeasurementDataReader,
    Measurement, DataTypes, Action, to_ts_data
)
from seglearn.base import TS_Data
from seglearn.pipe import Pype
from fhgutils import (
    Segment, contextual_recarray_dtype, filter_ts_data, filter_labels,
    one_label_per_window
)
from scipy.signal import resample
from collections import Counter
import librosa
import matplotlib.pyplot as plt


class ToolTrackingDataLoader:
    def __init__(self, source, window_length=0.4, overlap=0.5):
        self.source = source
        self.window_length = window_length
        self.overlap = overlap
        self.mdr = MeasurementDataReader(source=self.source)

    def load_measurement_data(self, tool):
        return self.mdr.query().filter_by(Tool == tool).get()

    def segment_data(self, data_dict):
        Xt, Xc, y = to_ts_data(data_dict, contextual_recarray_dtype)
        X = TS_Data(Xt, Xc)
        pipe = Pype(
            [
                (
                    'segment',
                    Segment(
                        window_length=self.window_length,
                        overlap=self.overlap,
                        enforce_size=True,
                        n=len(np.unique(Xc.desc))
                    )
                )
            ]
        )
        return pipe.fit_transform(X, y)

    def extract_mfcc_features(self, Xt):
        """Extract MFCC features for each window of microphone data."""
        print("[INFO] Extracting MFCC features for microphone")
        mfcc_features = []
        for window in Xt:
            signal = window.squeeze(axis=-1)  # flatten to 1D
            mfcc = librosa.feature.mfcc(y=signal, sr=8000, n_mfcc=1, hop_length=80, n_fft=400)
            mfcc_features.append(mfcc.T)  # transpose to (frames, n_mfcc)
        return np.array(mfcc_features)

    def filter_and_stack_sequences(self, Xt, Xc, y):
        """
        Filters Xt to retain only sequences with the most common shape,
        then filters Xc and y accordingly, and stacks Xt.

        Args:
            Xt (list of np.ndarray): List of time series windows (sequences).
            Xc (pd.DataFrame, pd.Series, or np.ndarray): Context or metadata for each sequence.
            y (np.ndarray or list): Target labels for each sequence.

        Returns:
            Xt_stacked (np.ndarray): Stacked 3D numpy array of sequences.
            Xc_filtered: Filtered Xc, same length as Xt_stacked.
            y_filtered: Filtered y, same length as Xt_stacked.
        """
        # Step 1: Count shapes
        shape_counts = Counter([x.shape for x in Xt])
        target_shape = shape_counts.most_common(1)[0][0]
        print(f"[Info] Target shape for stacking: {target_shape}")

        # Step 2: Filter Xt
        Xt_filtered = [x for x in Xt if x.shape == target_shape]

        # Step 3: Filter Xc and y by index
        valid_indices = [i for i, x in enumerate(Xt) if x.shape == target_shape]

        # Step 4: Handle Xc (DataFrame, Series, or ndarray)
        if hasattr(Xc, 'iloc'):
            Xc_filtered = Xc.iloc[valid_indices]
        else:
            Xc_filtered = Xc[valid_indices]

        # Step 5: Handle y
        y_filtered = np.array([y[i] for i in valid_indices])

        # Step 6: Stack Xt
        Xt_stacked = np.stack(Xt_filtered)

        # Optional: Warn if data was filtered
        if len(Xt_filtered) != len(Xt):
            print(f"[Warning] Dropped {len(Xt) - len(Xt_filtered)} sequences with non-matching shapes.")

        return Xt_stacked, Xc_filtered, y_filtered

        # ===== Visualization helpers =====

    def plot_raw_vs_downsampled(self, raw_windows, downsampled_windows, sensor_name, window_idx=0):
        t_raw = np.arange(raw_windows[window_idx].shape[0])
        t_down = np.linspace(0, raw_windows[window_idx].shape[0] - 1, downsampled_windows[window_idx].shape[0])

        plt.figure(figsize=(12, 6))
        n_features = raw_windows[window_idx].shape[1]
        for feat_idx in range(min(n_features, 3)):
            plt.subplot(min(n_features, 3), 1, feat_idx + 1)
            plt.plot(t_raw, raw_windows[window_idx][:, feat_idx], label="Raw")
            plt.plot(t_down, downsampled_windows[window_idx][:, feat_idx], label="Downsampled", linestyle='--')
            plt.title(f"{sensor_name} Feature {feat_idx} Window {window_idx}")
            plt.xlabel("Time (samples)")
            plt.ylabel("Value")
            plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_fused_sensors(self, Xt_fused, sensor_dims, window_idx=0):
        plt.figure(figsize=(15, 3 * len(sensor_dims)))
        start = 0
        for i, (sensor, dim) in enumerate(sensor_dims.items()):
            end = start + dim
            plt.subplot(len(sensor_dims), 1, i + 1)
            plt.plot(Xt_fused[window_idx, :, start:end])
            plt.title(f"Fused Sensor: {sensor}")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            start = end
        plt.tight_layout()
        plt.show()

    def plot_raw_vs_fused(self, raw_Xt_list, Xt_fused, window_index=0):
        """
        Compares raw sensor data windows vs fused data window for the same index.

        Args:
            raw_Xt_list (list of np.ndarray): List of raw segmented data arrays, one per sensor
                                              Each element shape: (num_windows, window_length_raw, features)
            Xt_fused (np.ndarray): Fused data array (num_windows, window_length_fused, total_features)
            window_index (int): Index of the window to compare
        """
        feature_counts = {
            'acc': 3,
            'gyr': 3,
            'mag': 3,
            'mic': 1,
        }
        sensors = list(feature_counts.keys())
        indices = np.cumsum([0] + list(feature_counts.values()))

        num_sensors = len(sensors)
        plt.figure(figsize=(15, 4 * num_sensors))

        time_fused = np.arange(Xt_fused.shape[1])

        for i, sensor in enumerate(sensors):
            # Raw data for this sensor
            raw_data = raw_Xt_list[i][window_index]
            time_raw = np.arange(raw_data.shape[0])

            # Fused data slice for this sensor
            start_idx = indices[i]
            end_idx = indices[i + 1]
            fused_data = Xt_fused[window_index, :, start_idx:end_idx]

            plt.subplot(num_sensors, 2, 2 * i + 1)
            for f in range(raw_data.shape[1]):
                plt.plot(time_raw, raw_data[:, f], label=f'{sensor}_feat{f}')
            plt.title(f'Raw {sensor} data - window {window_index}')
            plt.xlabel('Sample Index')
            plt.ylabel('Signal')
            plt.legend()

            plt.subplot(num_sensors, 2, 2 * i + 2)
            for f in range(fused_data.shape[1]):
                plt.plot(time_fused, fused_data[:, f], label=f'{sensor}_feat{f}')
            plt.title(f'Fused {sensor} data - window {window_index}')
            plt.xlabel('Sample Index')
            plt.ylabel('Signal')
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_fused_window(self, Xt_fused, window_index=0):
        """
        Plots the fused sensor data for a single window to verify timestamp alignment visually.

        Args:
            Xt_fused (np.ndarray): Fused data array of shape (num_windows, window_length, total_features)
            window_index (int): Index of the window to plot
        """
        feature_counts = {
            'acc': 3,
            'gyr': 3,
            'mag': 3,
            'mic': 1,
        }
        indices = np.cumsum([0] + list(feature_counts.values()))
        sensors = list(feature_counts.keys())

        plt.figure(figsize=(12, 8))
        time = np.arange(Xt_fused.shape[1])  # sample indices as proxy for time

        for i, sensor in enumerate(sensors):
            start_idx = indices[i]
            end_idx = indices[i + 1]
            sensor_data = Xt_fused[window_index, :, start_idx:end_idx]

            plt.subplot(len(sensors), 1, i + 1)
            for feature_idx in range(sensor_data.shape[1]):
                plt.plot(time, sensor_data[:, feature_idx], label=f'{sensor}_feat{feature_idx}')
            plt.title(f'Sensor: {sensor} - Window {window_index}')
            plt.xlabel('Sample Index')
            plt.ylabel('Signal')
            plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    def print_data_shapes(self, windowed_data_dict):
        print("Data shape summary:")
        for sensor, windows in windowed_data_dict.items():
            print(
                f"  {sensor}: {len(windows)} windows, window shape: {windows[0].shape if len(windows) > 0 else 'N/A'}")

    def process_segmented_data(self, X_trans, y_trans, desc_filters=None):

        if not desc_filters or 'all' in desc_filters:
            desc_filters = ['acc', 'gyr', 'mag', 'mic']

        print(f"[INFO] extract segmented {desc_filters} data")

        Xt_list = []
        max_length = 0
        raw_Xt_dict = {}
        raw_Xt_list = []

        for desc_filter in desc_filters:
            print(f"[INFO] Extracting segmented data for filter: {desc_filter}")
            Xt, Xc, y = filter_ts_data(X_trans, y_trans, filt={'desc': desc_filter})
            Xt, Xc, y = self.filter_and_stack_sequences(Xt, Xc, y)
            Xt = Xt[:, :, 1:]  # Drop timestamp (first column)

            raw_Xt_dict[desc_filter] = Xt.copy()  # Store raw pre-downsample for visualization

            raw_Xt_list.append(Xt)

            # Special processing for microphone
            if desc_filter == 'mic':
                Xt = self.extract_mfcc_features(Xt)


            # For magnetometer, downsample to 102 Hz if needed
            elif desc_filter == 'mag':
                print("[INFO] Downsampling magnetometer")
                target_length = 41
                downsampled = np.array([resample(window, target_length, axis=0) for window in Xt])
                # self.plot_raw_vs_downsampled(raw_Xt_dict[desc_filter], downsampled, desc_filter)
                Xt = downsampled

            # For acc, gyr, or other sensors, keep or resample to fixed length (say 41)
            else:
                target_length = 41
                downsampled = np.array([resample(window, target_length, axis=0) for window in Xt])
                # self.plot_raw_vs_downsampled(raw_Xt_dict[desc_filter], downsampled, desc_filter)
                Xt = downsampled


            Xt_f, _, y_f = filter_labels(labels=[-1], Xt=Xt, Xc=Xc, y=y)
            Xt_list.append(Xt_f)
            max_length = max(max_length, Xt_f.shape[1])

        # self.print_data_shapes({desc: Xt for desc, Xt in zip(desc_filters, Xt_list)})

        # Step 2: Align window counts
        min_windows = min([Xt.shape[0] for Xt in Xt_list])  # Minimum number of windows across all sensors

        print(f"[INFO] Aligning all sensors to {min_windows} windows")

        aligned_Xt_list = []
        y_f = y_f[:min_windows]  # Truncate labels to match min windows

        for Xt in Xt_list:
            Xt = Xt[:min_windows]  # Truncate to min window count
            aligned_Xt_list.append(Xt)

        # Step 3: resample each to max_length
        resampled_Xt_list = []
        for Xt_f in aligned_Xt_list:
            resampled = np.array([resample(window, max_length, axis=0) for window in Xt_f])
            resampled_Xt_list.append(resampled)

        # Fuse sensor data horizontally
        Xt_fused = np.concatenate(resampled_Xt_list, axis=-1)
        y_f = one_label_per_window(y=y_f)

        # Visualize fused sensors for first window
        # Define sensor dims here; adjust based on your features per sensor after processing
        sensor_dims = {
            'acc': 3,
            'gyr': 3,
            'mag': 3,
            'mic': 1  # MFCC outputs 1 feature in this example
        }
        # self.plot_fused_sensors(Xt_fused, sensor_dims)
        # self.plot_raw_vs_fused(raw_Xt_list, Xt_fused, window_index=65)

        return Xt_fused, y_f

    def load_and_process(self, tool, sensors):
        data_dict = self.load_measurement_data(tool)
        classes = data_dict["01"]['classes']
        X_trans, y_trans = self.segment_data(data_dict)
        Xt, y = self.process_segmented_data(X_trans, y_trans, sensors)
        return np.stack(Xt), np.stack(y), classes
