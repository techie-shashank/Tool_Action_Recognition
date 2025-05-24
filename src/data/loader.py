import numpy as np
from datatools import (
    Tool, Config, MeasurementSeries, MeasurementDataReader,
    Measurement, DataTypes, Action, to_ts_data
)
from seglearn.base import TS_Data
from seglearn.pipe import Pype
from fhgutils import Segment, contextual_recarray_dtype, filter_ts_data


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

    def filter_segmented_data(self, X_trans, y_trans, desc_filter):
        print(f"[INFO] extract segmented {desc_filter} data")
        Xt, Xc, y = filter_ts_data(X_trans, y_trans, filt={'desc': desc_filter})
        return Xt, Xc, y

    def load_and_process(self, tool, desc_filter):
        data_dict = self.load_measurement_data(tool)
        classes = data_dict["01"]['classes']
        X_trans, y_trans = self.segment_data(data_dict)
        Xt, Xc, y = self.filter_segmented_data(X_trans, y_trans, desc_filter)
        return np.stack(Xt), np.stack(Xc), np.stack(y), classes