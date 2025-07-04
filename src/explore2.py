import matplotlib.pyplot as plt
import pandas as pd
from data.loader import ToolTrackingDataLoader
import numpy as np  
# Configuration
SOURCE_PATH = r"C:\Users\taimo\Desktop\FAU-Courses\ADLTS-Seminar\data\tool-tracking-data"
TOOLS = ['electric_screwdriver', 'pneumatic_rivet_gun', 'pneumatic_screwdriver']
SENSORS = ['acc', 'gyr','mag','mic']

# def get_tool_sample_distribution():
#     """Calculate total samples per tool and per sensor"""
#     tool_samples = {}
#     sensor_samples = {}

#     for tool in TOOLS:
#         loader = ToolTrackingDataLoader(source=SOURCE_PATH)
#         data_dict = loader.load_measurement_data(tool)
#         tool_total = 0
                
#         for rec_id, rec_data in data_dict.items():
#             if rec_id == 'classes':
#                 continue
                
#             for sensor, df in rec_data.items():
#                 if sensor == 'classes':
#                     continue
                    
#                 # Count samples per sensor per tool
#                 sensor_key = f"{tool}_{sensor}"
#                 sensor_samples.setdefault(sensor_key, 0)
#                 sensor_samples[sensor_key] += len(df) 
                
#                 # Count total per tool
#                 tool_total += len(df)
        
#         tool_samples[tool] = tool_total
    
#     return tool_samples, sensor_samples

# def plot_tool_sample_distribution(tool_samples, sensor_samples):
#     """Visualize sample distribution using tables and bar plots"""
#     # Table 1: Total samples per tool
#     tool_df = pd.DataFrame.from_dict(
#         tool_samples, 
#         orient='index',
#         columns=['Total Samples']
#     ).reset_index().rename(columns={'index': 'Tool'})
    
#     print("Tool Sample Distribution:")
#     print(tool_df.to_string(index=False))
#     print("\n" + "="*50 + "\n")
    
#     # Table 2: Samples per sensor per tool
#     sensor_df = pd.DataFrame(
#         [(k.split('_')[0], k.split('_')[1], v) for k, v in sensor_samples.items()],
#         columns=['Tool', 'Sensor', 'Samples']
#     )
    
#     print("Sensor Sample Distribution:")
#     print(sensor_df.to_string(index=False))
    
#     # Visualization 1: Bar plot - Samples per tool
#     plt.figure(figsize=(12, 6))
    
#     # Sort data for better visualization
#     sorted_tool_df = tool_df.sort_values('Total Samples', ascending=False)
    
#     # Create bar plot with proper axis handling
#     bars = plt.bar(
#         x=range(len(sorted_tool_df)),
#         height=sorted_tool_df['Total Samples'],
#         color='skyblue',
#         edgecolor='black',
#         alpha=0.7
#     )
    
#     # Set x-axis labels with proper formatting
#     plt.xticks(
#         ticks=range(len(sorted_tool_df)),
#         labels=[tool.replace('_', ' ').title() for tool in sorted_tool_df['Tool']],
#         rotation=45,
#         ha='right',
#         fontsize=10
#     )
    
#     # Add value labels on bars
#     for i, bar in enumerate(bars):
#         height = bar.get_height()
#         plt.text(
#             bar.get_x() + bar.get_width()/2,
#             height + max(sorted_tool_df['Total Samples']) * 0.01,
#             f'{int(height):,}',
#             ha='center',
#             va='bottom',
#             fontsize=9,
#             fontweight='bold'
#         )
    
#     plt.title('Total Samples per Tool', fontsize=16, fontweight='bold', pad=20)
#     plt.xlabel('Tool Type', fontsize=12, fontweight='bold')
#     plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
#     plt.grid(axis='y', alpha=0.3, linestyle='--')
    
#     # Format y-axis with thousand separators
#     plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
#     plt.tight_layout()
#     plt.savefig('tool_sample_distribution.png', dpi=300, bbox_inches='tight')
#     plt.show()


# def plot_label_distribution_per_tool():
#     """Generate label distribution plots for each tool"""
#     for tool in TOOLS:
#         loader = ToolTrackingDataLoader(source=SOURCE_PATH)
#         data_dict = loader.load_measurement_data(tool)
        
#         # Extract class mapping from any recording
#         tool_classes = None
#         for rec_id, rec_data in data_dict.items():
#             if isinstance(rec_data, dict) and 'classes' in rec_data:
#                 tool_classes = rec_data['classes']
#                 break
        
#         if not tool_classes:
#             print(f"[Warning] No class mapping found for {tool}")
#             tool_classes = {}

#         all_labels = []
        
#         # Aggregate labels from ALL recordings (not just the last one)
#         for rec_id, rec_data in data_dict.items():
#             if not isinstance(rec_data, dict):
#                 continue
                
#             for sensor, df in rec_data.items():
#                 if sensor == 'classes' or 'label' not in df.columns:
#                     continue
#                 all_labels.append(df['label'])

#         if not all_labels:
#             print(f"No labels found for {tool}")
#             continue

#         # Combine all label series
#         label_series = pd.concat(all_labels)
#         label_counts = label_series.value_counts().sort_index()

#         # Generate human-readable label names
#         label_names = []
#         for idx in label_counts.index:
#             class_name = tool_classes.get(int(idx), f"Class_{idx}")
#             formatted_name = class_name.replace('_', ' ').strip().title()
#             label_names.append(formatted_name)

#         # Plotting
#         plt.figure(figsize=(12, 6))

#         bars = plt.bar(
#             x=label_names,
#             height=label_counts.values,
#             color='lightcoral',
#             edgecolor='black',
#             alpha=0.7
#         )

#         plt.xticks(rotation=45, ha='right', fontsize=10)

#         for i, bar in enumerate(bars):
#             height = bar.get_height()
#             plt.text(
#                 bar.get_x() + bar.get_width()/2,
#                 height + max(label_counts.values) * 0.01,
#                 f'{int(height):,}',
#                 ha='center',
#                 va='bottom',
#                 fontsize=9,
#                 fontweight='bold'
#             )

#         plt.title(f'Label Distribution: {tool.replace("_", " ").title()}', 
#                   fontsize=16, fontweight='bold', pad=20)
#         plt.xlabel('Class Labels', fontsize=12, fontweight='bold')
#         plt.ylabel('Sample Count', fontsize=12, fontweight='bold')
#         plt.grid(axis='y', alpha=0.3, linestyle='--')
#         plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
#         plt.tight_layout()
#         plt.savefig(f'label_distribution_{tool}.png', dpi=300, bbox_inches='tight')
#         plt.show()

# def plot_label_distribution_per_tool_and_sensor():
#     """Generate label distribution plots for each tool and sensor combination"""
#     # Define sensors to analyze (excluding mic due to different dimensions)
#     SENSORS = ['acc', 'gyr', 'mag']
    
#     for tool in TOOLS:
#         loader = ToolTrackingDataLoader(source=SOURCE_PATH)
#         data_dict = loader.load_measurement_data(tool)
        
#         # Extract class mapping from any recording
#         tool_classes = None
#         for rec_id, rec_data in data_dict.items():
#             if isinstance(rec_data, dict) and 'classes' in rec_data:
#                 tool_classes = rec_data['classes']
#                 break
        
#         if not tool_classes:
#             print(f"[Warning] No class mapping found for {tool}")
#             tool_classes = {}
        
#         # Process each sensor separately
#         for sensor in SENSORS:
#             print(f"\nProcessing {tool} - {sensor}")
            
#             all_labels = []
#             measurement_info = []
            
#             # Aggregate labels from ALL recordings for this specific sensor
#             for rec_id, rec_data in data_dict.items():
#                 if not isinstance(rec_data, dict):
#                     continue
                
#                 sensor_samples = 0
#                 if sensor in rec_data and 'label' in rec_data[sensor].columns:
#                     df = rec_data[sensor]
#                     all_labels.append(df['label'])
#                     sensor_samples = len(df)
                    
#                 if sensor_samples > 0:
#                     measurement_info.append(f"{rec_id}: {sensor_samples:,}")

#             if not all_labels:
#                 print(f"No labels found for {tool} - {sensor}")
#                 continue

#             # Combine all label series
#             label_series = pd.concat(all_labels)
#             label_counts = label_series.value_counts().sort_index()
            
#             # Create measurement breakdown text
#             total_samples = sum(label_counts.values)
#             measurement_text = " + ".join(measurement_info)
#             breakdown_text = f"Total: {total_samples:,} samples ({measurement_text})"

#             # Generate human-readable label names
#             label_names = []
#             for idx in label_counts.index:
#                 class_name = tool_classes.get(int(idx), f"Class_{idx}")
#                 formatted_name = class_name.replace('_', ' ').strip().title()
#                 label_names.append(formatted_name)

#             # Plotting
#             plt.figure(figsize=(14, 8))

#             bars = plt.bar(
#                 x=label_names,
#                 height=label_counts.values,
#                 color='lightcoral',
#                 edgecolor='black',
#                 alpha=0.7
#             )

#             plt.xticks(rotation=45, ha='right', fontsize=10)

#             # Add count labels on bars
#             for i, bar in enumerate(bars):
#                 height = bar.get_height()
#                 plt.text(
#                     bar.get_x() + bar.get_width()/2,
#                     height + max(label_counts.values) * 0.01,
#                     f'{int(height):,}',
#                     ha='center',
#                     va='bottom',
#                     fontsize=9,
#                     fontweight='bold'
#                 )

#             plt.title(f'Label Distribution: {tool.replace("_", " ").title()} - {sensor.upper()}\n{breakdown_text}', 
#                       fontsize=14, fontweight='bold', pad=20)
#             plt.xlabel('Class Labels', fontsize=12, fontweight='bold')
#             plt.ylabel('Sample Count', fontsize=12, fontweight='bold')
#             plt.grid(axis='y', alpha=0.3, linestyle='--')
#             plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
#             plt.tight_layout()
#             plt.savefig(f'label_distribution_{tool}_{sensor}.png', dpi=300, bbox_inches='tight')
#             plt.show()


# def plot_sensor_sample_distribution():
#     """Generate sample distribution plots for each sensor type across all tools"""
#     SENSORS = ['acc', 'gyr', 'mag', 'mic']
    
#     # Dictionary to store samples per tool per sensor
#     sensor_tool_samples = {sensor: {} for sensor in SENSORS}
    
#     for tool in TOOLS:
#         loader = ToolTrackingDataLoader(source=SOURCE_PATH)
#         data_dict = loader.load_measurement_data(tool)
        
#         for sensor in SENSORS:
#             total_samples = 0
            
#             # Count samples for this sensor across all recordings
#             for rec_id, rec_data in data_dict.items():
#                 if not isinstance(rec_data, dict):
#                     continue
                    
#                 if sensor in rec_data and 'label' in rec_data[sensor].columns:
#                     df = rec_data[sensor]
#                     total_samples += len(df)
            
#             if total_samples > 0:
#                 sensor_tool_samples[sensor][tool] = total_samples
    
#     # Create visualization for each sensor
#     for sensor in SENSORS:
#         if not sensor_tool_samples[sensor]:
#             continue
            
#         # Sort tools by sample count
#         sorted_tools = sorted(sensor_tool_samples[sensor].items(), 
#                             key=lambda x: x[1], reverse=True)
        
#         tools = [tool.replace('_', ' ').title() for tool, _ in sorted_tools]
#         samples = [count for _, count in sorted_tools]
        
#         plt.figure(figsize=(12, 6))
        
#         bars = plt.bar(
#             x=tools,
#             height=samples,
#             color='skyblue',
#             edgecolor='black',
#             alpha=0.7
#         )
        
#         plt.xticks(rotation=45, ha='right', fontsize=10)
        
#         # Add value labels on bars
#         for i, bar in enumerate(bars):
#             height = bar.get_height()
#             plt.text(
#                 bar.get_x() + bar.get_width()/2,
#                 height + max(samples) * 0.01,
#                 f'{int(height):,}',
#                 ha='center',
#                 va='bottom',
#                 fontsize=9,
#                 fontweight='bold'
#             )
        
#         plt.title(f'Sample Distribution: {sensor.upper()} Sensor', 
#                   fontsize=16, fontweight='bold', pad=20)
#         plt.xlabel('Tool Type', fontsize=12, fontweight='bold')
#         plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
#         plt.grid(axis='y', alpha=0.3, linestyle='--')
#         plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
#         plt.tight_layout()
#         plt.savefig(f'sensor_sample_distribution_{sensor}.png', dpi=300, bbox_inches='tight')
#         plt.show()


# # if __name__ == "__main__":
# #     # 1. Overall tool sample distribution (original)
# #     tool_samples, sensor_samples = get_tool_sample_distribution()
# #     plot_tool_sample_distribution(tool_samples, sensor_samples)
    
# #     # 2. Sample distribution by sensor type
# #     plot_sensor_sample_distribution()
    
# #     # 3. Label distribution per tool and sensor
# #     plot_label_distribution_per_tool_and_sensor()


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# def plot_combined_label_distribution():
#     """Generate a combined label distribution plot for all tools with grouped bars"""
    
#     # Dictionary to store data for each tool
#     tool_data = {}
#     all_unique_labels = set()
    
#     # Collect data for each tool
#     for tool in TOOLS:
#         loader = ToolTrackingDataLoader(source=SOURCE_PATH)
#         data_dict = loader.load_measurement_data(tool)
        
#         # Extract class mapping from any recording
#         tool_classes = None
#         for rec_id, rec_data in data_dict.items():
#             if isinstance(rec_data, dict) and 'classes' in rec_data:
#                 tool_classes = rec_data['classes']
#                 break
        
#         if not tool_classes:
#             print(f"[Warning] No class mapping found for {tool}")
#             tool_classes = {}
        
#         all_labels = []
        
#         # Aggregate labels from ALL recordings
#         for rec_id, rec_data in data_dict.items():
#             if not isinstance(rec_data, dict):
#                 continue
            
#             for sensor, df in rec_data.items():
#                 if sensor == 'classes' or 'label' not in df.columns:
#                     continue
#                 all_labels.append(df['label'])
        
#         if not all_labels:
#             print(f"No labels found for {tool}")
#             continue
        
#         # Combine all label series and count
#         label_series = pd.concat(all_labels)
#         label_counts = label_series.value_counts().sort_index()
        
#         # Generate human-readable label names and store counts
#         tool_label_data = {}
#         for idx in label_counts.index:
#             class_name = tool_classes.get(int(idx), f"Class_{idx}")
#             formatted_name = class_name.replace('_', ' ').strip().title()
#             tool_label_data[formatted_name] = label_counts.loc[idx]
#             all_unique_labels.add(formatted_name)
        
#         tool_data[tool] = tool_label_data
    
#     # Convert to consistent format - ensure all tools have data for all labels
#     all_labels_sorted = sorted(all_unique_labels)
    
#     # Create data arrays for plotting
#     tool_names = list(tool_data.keys())
#     n_tools = len(tool_names)
#     n_labels = len(all_labels_sorted)
    
#     # Prepare data matrix
#     data_matrix = np.zeros((n_tools, n_labels))
#     for i, tool in enumerate(tool_names):
#         for j, label in enumerate(all_labels_sorted):
#             data_matrix[i, j] = tool_data[tool].get(label, 0)
    
#     # Set up the plot
#     fig, ax = plt.subplots(figsize=(15, 8))
    
#     # Set up bar positions
#     bar_width = 0.25
#     x = np.arange(n_labels)
    
#     # Colors for each tool
#     colors = ['lightcoral', 'lightblue', 'lightgreen']
    
#     # Plot bars for each tool
#     for i, tool in enumerate(tool_names):
#         tool_display_name = tool.replace("_", " ").title()
#         bars = ax.bar(
#             x + i * bar_width,
#             data_matrix[i],
#             bar_width,
#             label=tool_display_name,
#             color=colors[i % len(colors)],
#             edgecolor='black',
#             alpha=0.7
#         )
        
    
#     # Customize the plot
#     ax.set_xlabel('Class Labels', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
#     ax.set_title('Label Distribution Comparison Across All Tools', 
#                  fontsize=16, fontweight='bold', pad=20)
    
#     # Set x-axis labels
#     ax.set_xticks(x + bar_width)
#     ax.set_xticklabels(all_labels_sorted, rotation=45, ha='right', fontsize=10)
    
#     # Add legend outside the plot area to prevent overlap
#     ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    
#     # Add grid
#     ax.grid(axis='y', alpha=0.3, linestyle='--')
    
#     # Format y-axis
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
#     # Adjust layout and save
#     plt.tight_layout()
#     plt.savefig('combined_label_distribution.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# def plot_combined_label_distribution_no_mic():
#     """Generate a combined label distribution plot for all tools with grouped bars, excluding mic sensor data"""
    
#     # Dictionary to store data for each tool
#     tool_data = {}
#     all_unique_labels = set()
    
#     # Collect data for each tool
#     for tool in TOOLS:
#         loader = ToolTrackingDataLoader(source=SOURCE_PATH)
#         data_dict = loader.load_measurement_data(tool)
        
#         # Extract class mapping from any recording
#         tool_classes = None
#         for rec_id, rec_data in data_dict.items():
#             if isinstance(rec_data, dict) and 'classes' in rec_data:
#                 tool_classes = rec_data['classes']
#                 break
        
#         if not tool_classes:
#             print(f"[Warning] No class mapping found for {tool}")
#             tool_classes = {}
        
#         all_labels = []
        
#         # Aggregate labels from ALL recordings, excluding mic sensor
#         for rec_id, rec_data in data_dict.items():
#             if not isinstance(rec_data, dict):
#                 continue
            
#             for sensor, df in rec_data.items():
#                 if sensor == 'classes' or 'label' not in df.columns:
#                     continue
#                 # Skip mic sensor data
#                 if sensor.lower() == 'mic':
#                     continue
#                 all_labels.append(df['label'])
        
#         if not all_labels:
#             print(f"No labels found for {tool} (excluding mic)")
#             continue
        
#         # Combine all label series and count
#         label_series = pd.concat(all_labels)
#         label_counts = label_series.value_counts().sort_index()
        
#         # Generate human-readable label names and store counts
#         tool_label_data = {}
#         for idx in label_counts.index:
#             class_name = tool_classes.get(int(idx), f"Class_{idx}")
#             formatted_name = class_name.replace('_', ' ').strip().title()
#             tool_label_data[formatted_name] = label_counts.loc[idx]
#             all_unique_labels.add(formatted_name)
        
#         tool_data[tool] = tool_label_data
    
#     # Convert to consistent format - ensure all tools have data for all labels
#     all_labels_sorted = sorted(all_unique_labels)
    
#     # Create data arrays for plotting
#     tool_names = list(tool_data.keys())
#     n_tools = len(tool_names)
#     n_labels = len(all_labels_sorted)
    
#     # Prepare data matrix
#     data_matrix = np.zeros((n_tools, n_labels))
#     for i, tool in enumerate(tool_names):
#         for j, label in enumerate(all_labels_sorted):
#             data_matrix[i, j] = tool_data[tool].get(label, 0)
    
#     # Set up the plot
#     fig, ax = plt.subplots(figsize=(15, 8))
    
#     # Set up bar positions
#     bar_width = 0.25
#     x = np.arange(n_labels)
    
#     # Colors for each tool
#     colors = ['lightcoral', 'lightblue', 'lightgreen']
    
#     # Plot bars for each tool
#     for i, tool in enumerate(tool_names):
#         tool_display_name = tool.replace("_", " ").title()
#         bars = ax.bar(
#             x + i * bar_width,
#             data_matrix[i],
#             bar_width,
#             label=tool_display_name,
#             color=colors[i % len(colors)],
#             edgecolor='black',
#             alpha=0.7
#         )
        
#         # Value labels removed for cleaner visualization
    
#     # Customize the plot
#     ax.set_xlabel('Class Labels', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
#     ax.set_title('Label Distribution Comparison Across All Tools (Excluding Mic Sensor)', 
#                  fontsize=16, fontweight='bold', pad=20)
    
#     # Set x-axis labels
#     ax.set_xticks(x + bar_width)
#     ax.set_xticklabels(all_labels_sorted, rotation=45, ha='right', fontsize=10)
    
#     # Add legend outside the plot area to prevent overlap
#     ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    
#     # Add grid
#     ax.grid(axis='y', alpha=0.3, linestyle='--')
    
#     # Format y-axis
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
#     # Adjust layout and save
#     plt.tight_layout()
#     plt.savefig('combined_label_distribution_no_mic.png', dpi=300, bbox_inches='tight')
#     plt.show()
    
    
def get_tool_sample_distribution_no_mic():
    """Calculate total samples per tool and per sensor, excluding mic sensor"""
    tool_samples = {}
    sensor_samples = {}

    for tool in TOOLS:
        loader = ToolTrackingDataLoader(source=SOURCE_PATH)
        data_dict = loader.load_measurement_data(tool)
        tool_total = 0
                
        for rec_id, rec_data in data_dict.items():
            if rec_id == 'classes':
                continue
                
            for sensor, df in rec_data.items():
                if sensor == 'classes':
                    continue
                
                # Skip mic sensor data
                if sensor.lower() == 'mic':
                    continue
                    
                # Count samples per sensor per tool
                sensor_key = f"{tool}_{sensor}"
                sensor_samples.setdefault(sensor_key, 0)
                sensor_samples[sensor_key] += len(df) 
                
                # Count total per tool
                tool_total += len(df)
        
        tool_samples[tool] = tool_total
    
    return tool_samples, sensor_samples

def plot_tool_sample_distribution_no_mic(tool_samples, sensor_samples):
    """Visualize sample distribution using tables and bar plots, excluding mic sensor"""
    # Table 1: Total samples per tool
    tool_df = pd.DataFrame.from_dict(
        tool_samples, 
        orient='index',
        columns=['Total Samples']
    ).reset_index().rename(columns={'index': 'Tool'})
    
    print("Tool Sample Distribution (Excluding Mic Sensor):")
    print(tool_df.to_string(index=False))
    print("\n" + "="*50 + "\n")
    
    # Table 2: Samples per sensor per tool
    sensor_df = pd.DataFrame(
        [(k.split('_')[0], k.split('_')[1], v) for k, v in sensor_samples.items()],
        columns=['Tool', 'Sensor', 'Samples']
    )
    
    print("Sensor Sample Distribution (Excluding Mic Sensor):")
    print(sensor_df.to_string(index=False))
    
    # Visualization 1: Bar plot - Samples per tool
    plt.figure(figsize=(12, 6))
    
    # Sort data for better visualization
    sorted_tool_df = tool_df.sort_values('Total Samples', ascending=False)
    
    # Create bar plot with proper axis handling
    bars = plt.bar(
        x=range(len(sorted_tool_df)),
        height=sorted_tool_df['Total Samples'],
        color='skyblue',
        edgecolor='black',
        alpha=0.7
    )
    
    # Set x-axis labels with proper formatting
    plt.xticks(
        ticks=range(len(sorted_tool_df)),
        labels=[tool.replace('_', ' ').title() for tool in sorted_tool_df['Tool']],
        rotation=45,
        ha='right',
        fontsize=10
    )
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + max(sorted_tool_df['Total Samples']) * 0.01,
            f'{int(height):,}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    plt.title('Total Samples per Tool (Excluding Mic Sensor)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Tool Type', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Format y-axis with thousand separators
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.savefig('tool_sample_distribution_no_mic.png', dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    # 1. Tool sample distribution
    tool_samples, sensor_samples = get_tool_sample_distribution_no_mic()
    plot_tool_sample_distribution_no_mic(tool_samples, sensor_samples)
    
    # # 2. Label distribution per tool
    # plot_label_distribution_per_tool()
    # plot_combined_label_distribution()
    # plot_combined_label_distribution_no_mic()