import pandas as pd
from scipy import signal
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

def augment_dataset(file_path: str):
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

    # TODO: in this phase, while augmentig, we need to also create the new ActionNet_test and ActionNet_train that contains the new splits of the new augmented dataset
    # TODO: replace "Get items from refrigerator/cabinets/drawers" to "Get/replace items from refrigerator/cabinets/drawers" and "Open a jar of almond butter" to "Open/close a jar of almond butter" to obtain 20 labels

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
#             'start_timestamp': data.loc[i, 'start'], # TODO: remove t0
#             'stop_timestamp': data.loc[i, 'stop'], # TODO: remove t0
#             'start_frame': data.loc[i, 'start'] * 29.67, # TODO: remove t0
#             'stop_frame': data.loc[i, 'stop'] * 29.67, # TODO: remove t0
#             'verb': data.loc[i, 'description'],
#             'verb_class': emg_descriptions_to_labels.index(data.loc[i, 'description']),
#         }

#         new_data.append(new_row)

#     # Convert the list of dictionaries to a DataFrame
#     new_data = pd.DataFrame(new_data)

#     # Save the DataFrame to a pickle file
#     file_name, file_extension = os.path.splitext(file_path)
#     new_data.to_pickle(f'{file_name}_rgb.pkl')

""" 

1. Each channel is rectified by taking the absolute value
2. Low-pass filter with cutoff frequency 5 Hz is applied 
3. All 8 channels from an armband are then jointly normalized and shifted to the range [−1, 1] using the minimum and maximum values across all channels
4. The absolute value of EMG data across all 8 forearm channels are summed together in each timestep to indicate overall forearm activation
5. The streams are then smoothed to focus on low-frequency signals on time scales comparable to slicing motions (ACTUALLY REFERS TO THE PREVIOUS APPLIED FILTER ACCORDING TO SLACK)

"""

def map_to_range_linear(side):
    mapped_side = np.zeros_like(side)
    for col in range(side.shape[1]):
        col_min, col_max = np.min(side[:, col]), np.max(side[:, col])
        mapped_side[:, col] = (side[:, col] - col_min) / (col_max - col_min) * (1 - (-1)) + (-1)
    return mapped_side

def z_norm(side):
    mean, std = np.mean(side, axis=0), np.std(side, axis=0)
    to_return = map_to_range_linear((side - mean) / std)
    
    return to_return

def emg_adjust_features(file_path: str, *, cut_frequency: float = 5.0, filter_order: int = 4):
    data = pd.DataFrame(pd.read_pickle(file_path))
    
    tmp_lefts, tmp_rights = data.loc[1:, "myo_left_readings"], data.loc[1:, "myo_right_readings"]
    length_periods_l, length_periods_r = [len(p) for p in tmp_lefts], [len(p) for p in tmp_rights]

    fs = 160                    # sampling frequency
    nyq = 0.5 * fs              # nyquist 
    normalized_cutoff = cut_frequency / nyq    #normalized cutoff frequency

    _, filt_coeffs = signal.butter(filter_order, normalized_cutoff, btype='low')
    
    NUM_CHANNELS = 8
    
    def apply_low_pass_filter(myo_side_readings):
        myo_side_readings = np.array([channels_values for period in myo_side_readings for channels_values in period])
        myo_side_readings = np.absolute(myo_side_readings)
        filtered_data = np.zeros_like(myo_side_readings)
        for i in range(NUM_CHANNELS):
            filtered_data[:, i] = signal.filtfilt(filt_coeffs, [1], myo_side_readings[:, i])
        
        return filtered_data

    
    filtered_data_left, filtered_data_right = apply_low_pass_filter(tmp_lefts), apply_low_pass_filter(tmp_rights)
    filtered_data_left, filtered_data_right = z_norm(filtered_data_left), z_norm(filtered_data_right)
    filtered_data_left, filtered_data_right = map_to_range_linear(filtered_data_left), map_to_range_linear(filtered_data_right)
    
    def put_back_into_dataframe(side_name, preprocessed, lengths):
        start = 0
        for i, period_length in enumerate(lengths):
            aus = np.empty((0, 8))
            
            for l in range(period_length):
                aus =np.vstack((aus, preprocessed[start + l]))
            
            summed_channels = np.sum(aus, axis=0)
            
            #SUM EACH CHANNEL FOR EACH PERIOD
            data.at[i+1, side_name] = summed_channels.tolist()
            
            start += period_length
    
    put_back_into_dataframe("myo_left_readings", filtered_data_left, length_periods_l)
    put_back_into_dataframe("myo_right_readings", filtered_data_right, length_periods_r)
    
    return data

if __name__ == '__main__':
    pass

# TODO: spectogram
# TODO: saples are not balanced maybe
