import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

class MultiSensorVisualizer:
    def __init__(self, base_path, tool='pneumatic_rivet_gun'):
        # Initialize base path and tool
        self.base_path = base_path
        self.tool = tool
        self.tools_data = self._get_tool_paths()

        if self.tool not in self.tools_data:
            raise ValueError(f"Tool '{self.tool}' not supported or data paths not set.")

        self.sensors_files = self.tools_data[self.tool]['sensors']
        self.annotation_file = self.tools_data[self.tool]['annotation']

        # Action ID to name mapping
        self.action_names = {
            0: 'undefined',
            1: 'tightening',
            2: 'untightening',
            3: 'motor_activity_cw',
            4: 'motor_activity_ccw',
            5: 'manual_motor_rotation',
            6: 'shaking',
            7: 'tightening_double'
        }

        # Colors for plotting actions
        self.action_colors = {
            0: '#D3D3D3',
            1: '#1F77B4',
            2: '#FF7F0E',
            3: '#2CA02C',
            4: '#D62728',
            5: '#9467BD',
            6: '#8C564B',
            7: '#E377C2'
        }

        self.sensor_data = {}
        self.annotation_data = None

    def _get_tool_paths(self):
        return {
            'pneumatic_rivet_gun': {
                'sensors': {
                    'ACC': os.path.join(self.base_path, 'pneumatic_rivet_gun', 'pythagoras-07-20200724', 'ACC-01-102.290.csv'),
                    'GYR': os.path.join(self.base_path, 'pneumatic_rivet_gun', 'pythagoras-07-20200724', 'GYR-01-102.290.csv'),
                    'MAG': os.path.join(self.base_path, 'pneumatic_rivet_gun', 'pythagoras-07-20200724', 'MAG-01-154.966.csv'),
                    'MIC': os.path.join(self.base_path, 'pneumatic_rivet_gun', 'pythagoras-07-20200724', 'MIC-01-8000.csv')
                },
                'annotation': os.path.join(self.base_path, 'pneumatic_rivet_gun', 'pythagoras-07-20200724', 'data-01.annotation~')
            }
        }

    def load_sensor_data(self):
        for sensor, path in self.sensors_files.items():
            if not os.path.exists(path):
                print(f"Warning: Sensor file {path} does not exist.")
                continue
            try:
                if sensor == 'MIC':
                    df = pd.read_csv(path, sep=';', header=None, names=['time', 'mic_val'],
                                     dtype={'time': float, 'mic_val': float}, on_bad_lines='skip')
                    df['time_rel'] = df['time'] - df['time'].iloc[0]
                    self.sensor_data[sensor] = df
                else:
                    df = pd.read_csv(path, sep=';', skiprows=1,
                                     names=['time', f'{sensor.lower()}_x', f'{sensor.lower()}_y', f'{sensor.lower()}_z'],
                                     dtype={'time': float,
                                            f'{sensor.lower()}_x': float,
                                            f'{sensor.lower()}_y': float,
                                            f'{sensor.lower()}_z': float},
                                     on_bad_lines='skip')
                    df['time_rel'] = df['time'] - df['time'].iloc[0]
                    df[f'{sensor.lower()}_mag'] = np.sqrt(
                        df[f'{sensor.lower()}_x']**2 +
                        df[f'{sensor.lower()}_y']**2 +
                        df[f'{sensor.lower()}_z']**2
                    )
                    self.sensor_data[sensor] = df
            except Exception as e:
                print(f"Error loading sensor {sensor} from {path}: {e}")

    def load_annotations(self):
        if not os.path.exists(self.annotation_file):
            print(f"Warning: Annotation file {self.annotation_file} does not exist.")
            return

        try:
            self.annotation_data = pd.read_csv(
                self.annotation_file,
                sep=';',
                header=None,
                names=['start', 'end', 'action', 'unknown'],
                dtype={'start': float, 'end': float, 'action': int, 'unknown': str},
                on_bad_lines='skip'
            )
            self.annotation_data['start_rel'] = self.annotation_data['start']
            self.annotation_data['end_rel'] = self.annotation_data['end']
        except Exception as e:
            print(f"Error loading annotations: {e}")
            self.annotation_data = None

    def plot(self):
        if not self.sensor_data:
            print("No sensor data to plot.")
            return
        if self.annotation_data is None or self.annotation_data.empty:
            print("No annotation data loaded; plotting sensor data without action overlays.")

        num_sensors = len(self.sensor_data)
        fig, axs = plt.subplots(num_sensors, 1, figsize=(15, 4 * num_sensors), sharex=True)
        if num_sensors == 1:
            axs = [axs]

        for ax, (sensor, df) in zip(axs, self.sensor_data.items()):
            if sensor == 'MIC':
                ax.plot(df['time_rel'], df['mic_val'], color='black', label='MIC signal')
                ylabel = 'MIC amplitude'
            else:
                mag_col = f'{sensor.lower()}_mag'
                ax.plot(df['time_rel'], df[mag_col], color='black', label=f'{sensor} magnitude')
                ylabel = f'{sensor} magnitude'

            if self.annotation_data is not None and not self.annotation_data.empty:
                for _, row in self.annotation_data.iterrows():
                    mask = (df['time_rel'] >= row['start_rel']) & (df['time_rel'] <= row['end_rel'])
                    segment = df.loc[mask]
                    if not segment.empty:
                        ax.plot(
                            segment['time_rel'],
                            segment[mag_col] if sensor != 'MIC' else segment['mic_val'],
                            color=self.action_colors.get(row['action'], '#000000'),
                            linewidth=2.5,
                            label=self.action_names.get(row['action'], f'Action {row["action"]}')
                        )

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), title='Action / Sensor')
            ax.set_ylabel(ylabel)
            ax.grid(True)

        axs[-1].set_xlabel('Time (s)')
        plt.suptitle('Multi-Sensor Data with Action Annotations')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize multi-sensor data with action annotations.')
    parser.add_argument('--base_path', type=str, required=True, help='Base path to the tool-tracking dataset.')
    parser.add_argument('--tool', type=str, default='pneumatic_rivet_gun', help='Tool name (default: pneumatic_rivet_gun)')
    args = parser.parse_args()

    visualizer = MultiSensorVisualizer(base_path=args.base_path, tool=args.tool)
    visualizer.load_sensor_data()
    visualizer.load_annotations()
    visualizer.plot()

if __name__ == '__main__':
    main()
