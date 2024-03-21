import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def chunk_timestamps_and_readings(left_timestamps, left_readings, right_timestamps, right_readings):
    # Initialize empty lists to store the chunks and readings
    left_chunks = [[]]
    right_chunks = [[]]
    left_reading_chunks = [[]]
    right_reading_chunks = [[]]

    # Initialize variables to track the current chunk start and end
    left_chunk_start = left_timestamps[0]
    left_chunk_end = left_chunk_start + 5
    right_chunk_start = right_timestamps[0]
    right_chunk_end = right_chunk_start + 5
    
    # Iterate over each time in both left and right lists
    for left_time, left_reading in zip(left_timestamps, left_readings):
        # Check if the left time is within the current left chunk
        if left_time < left_chunk_end:
            left_chunks[-1].append(left_time)
            left_reading_chunks[-1].append(left_reading)
        else:
            # Move to the next left chunk
            left_chunks.append([left_time])
            left_reading_chunks.append([left_reading])
            left_chunk_start = left_time
            left_chunk_end = left_chunk_start + 5
        
    # Iterate over each time in both right and right lists
    for right_time, right_reading in zip(right_timestamps, right_readings):
        # Check if the right time is within the current right chunk
        if right_time < right_chunk_end:
            right_chunks[-1].append(right_time)
            right_reading_chunks[-1].append(right_reading)
        else:
            # Move to the next right chunk
            right_chunks.append([right_time])
            right_reading_chunks.append([right_reading])
            right_chunk_start = right_time
            right_chunk_end = right_chunk_start + 5
    
    return left_chunks, right_chunks, left_reading_chunks, right_reading_chunks

def increment_samples(file_path: str):
    data = pd.DataFrame(pd.read_pickle(file_path))
    new_data = []

    for i, _ in data.iterrows():
        if i == 0:
            continue
        left_timestamps_chunks, right_timestamps_chunks, left_readings_chunks, right_readings_chunks = chunk_timestamps_and_readings(data.loc[i, 'myo_left_timestamps'], data.loc[i, 'myo_left_readings'], data.loc[i, 'myo_right_timestamps'], data.loc[i, 'myo_right_readings'])

        if len(left_timestamps_chunks) < len(right_timestamps_chunks):
            right_timestamps_chunks = right_timestamps_chunks[:-1]
            right_readings_chunks = right_readings_chunks[:-1]
        if len(left_timestamps_chunks) > len(right_timestamps_chunks):
            left_timestamps_chunks = left_timestamps_chunks[:-1]
            left_readings_chunks = left_readings_chunks[:-1]

        n_chunks = len(left_timestamps_chunks)

        for c in range(n_chunks):
            new_row = {
                'description': data.loc[i, 'description'],
                'start': min(left_timestamps_chunks[c][0], right_timestamps_chunks[c][0]),
                'stop': min(left_timestamps_chunks[c][-1], right_timestamps_chunks[c][-1]),
                'myo_left_timestamps': left_timestamps_chunks[c],
                'myo_left_readings': left_readings_chunks[c],
                'myo_right_timestamps': right_timestamps_chunks[c],
                'myo_right_readings': right_readings_chunks[c]
                }
            new_data.append(new_row)

    # Convert the list of dictionaries to a DataFrame
    new_data = pd.DataFrame(new_data)

    # Save the DataFrame to a pickle file
    file_name, file_extension = os.path.splitext(file_path)
    new_data.to_pickle(f'{file_name}_augmented.pkl')

# TODO: code to be used to prepare spilt train and test for rgb action net
# def setup_pickle_rgb(file_path: str):
#     data = pd.DataFrame(pd.read_pickle(file_path))
#     new_data = []

#     for i, _ in data.iterrows():
#         new_row = {
#             'uid': i,
#             'participant_id': 'P04',
#             'video_id': 'P04_01',
#             'narration': data.loc[i, 'description'],
#             'start_timestamp': data.loc[i, 'start'],
#             'stop_timestamp': data.loc[i, 'stop'],
#             'start_frame': data.loc[i, 'start'] * 29.67,
#             'stop_frame': data.loc[i, 'stop'] * 29.67,
#             'verb': data.loc[i, 'description'],
#             'verb_class': emg_descriptions_to_labels.index(data.loc[i, 'description']),
#         }

#         new_data.append(new_row)

#     # Convert the list of dictionaries to a DataFrame
#     new_data = pd.DataFrame(new_data)

#     # Save the DataFrame to a pickle file
#     file_name, file_extension = os.path.splitext(file_path)
#     new_data.to_pickle(f'{file_name}_rgb.pkl')


def emg_adjust_features(file_path: str, *, cut_frequency: float = 5.0, filter_order: int = 4):
    data = pd.DataFrame(pd.read_pickle(file_path))
    to_read = ['myo_right', 'myo_left']

    # Absolute value of the readings and application of low pass filter
    for side in to_read:
        for i, _ in data.iterrows():
            if i != 0:
                # Rectify channels
                row = abs(data.loc[i, side + '_readings'])
                times = data.loc[i, side + '_timestamps']
                
                # Low-pass filter 5 Hz
                fs = times.size / (times[-1]-times[0])

                # TODO: why is fs not 160? sometimes is more sometimes is less
                # TODO: maybe onw just filter with fs=160 is enough

                nyq = 0.5 * fs
                wn = cut_frequency / nyq

                b, a = butter(filter_order, wn, 'low', analog=False)
                filtered = filtfilt(b, a, row.T, padlen=1).T

                data.at[i, side + '_readings'] = filtered

    # TODO: how to do normalization ? is L2 norm ok ? is the normalization per each action or on the whole dataset / we can also use batch norm layer
    # TODO: use the statistics calculated on training to the testing phase
    # Normalization with L2 norm
    for side in to_read:
        flat_side_data = [it for sub in data.loc[1:, side + '_readings'].apply(lambda x: x) for rd in sub for it in rd]
        l2norm = np.linalg.norm(flat_side_data)
        
        for i, _ in data.iterrows():
            if i != 0:
                row = data.loc[i, side + '_readings']
                normalized = row / l2norm
                
                data.at[i, side + '_readings'] = normalized
    
    # Shift of the value range in [-1, 1]
    for side in to_read:
        flat_side_data = [it for sub in data.loc[1:, side + '_readings'].apply(lambda x: x) for rd in sub for it in rd]
        mx = max(flat_side_data)
        mn = min(flat_side_data)
        
        for i, _ in data.iterrows():
            if i != 0:
                row = data.loc[i, side + '_readings']
                # Normalize and shift in [-1, 1]
                normalized = (row - mn) * 2 / (mx-mn) - 1
                
                # TODO: avoid summation because input of lstm is 100x16 instead of 100x1 (slack confirmed 5th february)
                # Forearm activation
                activation = normalized.sum(1)
                
                data.at[i, side + '_readings'] = activation
        
    return data

def emg_adjust_features_index(file_path: str, index: int, *, cut_frequency: float = 5.0, filter_order: int = 4):
    data = pd.DataFrame(pd.read_pickle(file_path))
    to_read = ['myo_right', 'myo_left']

    transformed = {}

    for side in to_read:
        # Rectify channels
        row = abs(data.loc[index, side + '_readings'])
        times = data.loc[index, side + '_timestamps']
        
        # Low-pass filter 5 Hz
        fs = times.size / (times[-1]-times[0])

        nyq = 0.5 * fs
        wn = cut_frequency / nyq
        
        b, a = butter(filter_order, wn, 'low', analog=False)
        filtered = filtfilt(b, a, row.T).T
        
        # Normalize in [-1, 1]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        filtered = scaler.fit_transform(filtered)
        
        # Forearm activation
        activation = filtered.sum(1)
        
        transformed[side + '_readings'] = activation
        
    transformed['description'] = data.loc[index, 'description']
    
    return transformed

if __name__ == '__main__':
    setup_pickle_rgb('emg/S04_1_augmented.pkl')
            

# TODO: augmentation
# TODO: read sepctogram, why?
# TODO: divide each sample in 5 seconds segment
# TODO: saples are not balanced maybe
# TODO: split train and test must be modified for emg

# frame_rate = 108771/3385
# 14m18s
# 56m24s
# frame_rate = 29.67