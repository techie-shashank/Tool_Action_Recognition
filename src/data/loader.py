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

    def process_segmented_data(self, X_trans, y_trans, desc_filters=None):

        if not desc_filters or 'all' in desc_filters:
            desc_filters = ['acc', 'gyr', 'mag', 'mic']

        print(f"[INFO] extract segmented {desc_filters} data")

        Xt_list = []
        max_length = 0

        for desc_filter in desc_filters:
            print(f"[INFO] Extracting segmented data for filter: {desc_filter}")
            Xt, Xc, y = filter_ts_data(X_trans, y_trans, filt={'desc': desc_filter})
            Xt, Xc, y = self.filter_and_stack_sequences(Xt, Xc, y)
            Xt = Xt[:, :, 1:]  # Drop timestamp (first column)

            # Special processing for microphone
            if desc_filter == 'mic':
                Xt = self.extract_mfcc_features(Xt)

            # For magnetometer, downsample to 102 Hz if needed
            elif desc_filter == 'mag':
                print("[INFO] Downsampling magnetometer")
                target_length = 62  # example window length for acc/gyr
                Xt = np.array([resample(window, target_length, axis=0) for window in Xt])

            # For acc, gyr, or other sensors, keep or resample to fixed length (say 41)
            else:
                target_length = 62
                Xt = np.array([resample(window, target_length, axis=0) for window in Xt])


            Xt_f, _, y_f = filter_labels(labels=[-1], Xt=Xt, Xc=Xc, y=y)
            Xt_list.append(Xt_f)
            max_length = max(max_length, Xt_f.shape[1])

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
        return Xt_fused, y_f

    def load_and_process(self, tool, sensors):
        data_dict = self.load_measurement_data(tool)
        classes = data_dict["01"]['classes']
        X_trans, y_trans = self.segment_data(data_dict)
        Xt, y = self.process_segmented_data(X_trans, y_trans, sensors)
        return np.stack(Xt), np.stack(y), classes
