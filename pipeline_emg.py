import pandas as pd
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa

emg_descriptions_to_labels = [
    'Clean a pan with a sponge',
    'Clean a pan with a towel',
    'Clean a plate with a sponge',
    'Clean a plate with a towel',
    'Clear cutting board',
    'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
    'Get/replace items from refrigerator/cabinets/drawers',
    'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
    'Open/close a jar of almond butter',
    'Peel a cucumber',
    'Peel a potato',
    'Pour water from a pitcher into a glass',
    'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
    'Slice a cucumber',
    'Slice a potato',
    'Slice bread',
    'Spread almond butter on a bread slice',
    'Spread jelly on a bread slice',
    'Stack on table: 3 each large/small plates, bowls',
    'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
]

emg_descriptions_conversion_dict = {
            'Get items from refrigerator/cabinets/drawers'         :       'Get/replace items from refrigerator/cabinets/drawers',
            'Open a jar of almond butter'                          :       'Open/close a jar of almond butter'
        }

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

def create_split_augmented(old_file_path: str, new_file_path: str):
    data = pd.DataFrame(pd.read_pickle(new_file_path))
    if os.path.isfile('action-net/ActionNet_test_augmented.pkl'):
        test = pd.DataFrame(pd.read_pickle('action-net/ActionNet_test_augmented.pkl'))
    else:
        test = pd.DataFrame(pd.read_pickle('action-net/ActionNet_test.pkl'))

    if os.path.isfile('action-net/ActionNet_train_augmented.pkl'):
        train = pd.DataFrame(pd.read_pickle('action-net/ActionNet_train_augmented.pkl'))
    else:
        train = pd.DataFrame(pd.read_pickle('action-net/ActionNet_train.pkl'))

    old_file_name = old_file_path.split('/')[1]
    new_file_name = new_file_path.split('/')[1]

    # mapping each old index to the list of new (augmented) indices
    mapping_new_index = {}
    for i, _ in data.iterrows():
        old_index = data.loc[i, 'old_index']
        new_index = i
        
        if old_index in mapping_new_index:
            mapping_new_index[old_index].append(new_index)
        else:
            mapping_new_index[old_index] = [new_index]

    # TEST: creating the new rows of the new file split that refer to the augmented dataset
    old_test_rows = test[test['file'] == old_file_name]
    new_test_rows = []
    new_data = []

    for i, row in old_test_rows.iterrows():
        old_index = row['index']
        for new_index in mapping_new_index[old_index]:
            # handle old descriptions
            if row['description'] in emg_descriptions_conversion_dict.keys():
                new_description = emg_descriptions_conversion_dict[row['description']]
                old_test_rows.at[i, 'description'] = new_description

            new_row = {
                'index': int(new_index),
                'file': new_file_name,
                'description': row['description'] if row['description'] not in emg_descriptions_conversion_dict.keys() else emg_descriptions_conversion_dict[row['description']],
                'labels': row['labels']
            }
            new_data.append(new_row)

    new_test_rows = test[test['file'] != old_file_name]
    new_test_rows = pd.concat([new_test_rows, pd.DataFrame(new_data)])

    # TRAIN: creating the new rows of the new file split that refer to the augmented dataset
    old_train_rows = train[train['file'] == old_file_name]
    new_train_rows = []
    new_data = []

    for i, row in old_train_rows.iterrows():
        old_index = row['index']
        for new_index in mapping_new_index[old_index]:
            # handle old descriptions
            new_row = {
                'index': int(new_index),
                'file': new_file_name,
                'description': row['description'] if row['description'] not in emg_descriptions_conversion_dict.keys() else emg_descriptions_conversion_dict[row['description']],
                'labels': row['labels']
            }
            new_data.append(new_row)

    new_train_rows = train[train['file'] != old_file_name]
    new_train_rows = pd.concat([new_train_rows, pd.DataFrame(new_data)])

    new_test_rows.set_index('index', inplace=True, drop=False)
    new_train_rows.set_index('index', inplace=True, drop=False)

    # Save the DataFrame to a pickle file
    new_test_rows.to_pickle(f'action-net/ActionNet_test_augmented.pkl')
    new_train_rows.to_pickle(f'action-net/ActionNet_train_augmented.pkl')

    # TODO: create three columns: 'file_emg', 'file_rgb', 'file_specto' with the corresponding path to pkl files with the samples

def augment_partition(file_path: str):
    data = pd.DataFrame(pd.read_pickle(file_path))
    new_data = []

    for i, _ in data.iterrows():
        if data.loc[i, 'description'] == 'calibration':
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
            # handle old descriptions
            if data.loc[i, 'description'] in emg_descriptions_conversion_dict:
                data.at[i, 'description'] = emg_descriptions_conversion_dict[data.loc[i, 'description']]
            
            new_row = {
                'old_index': i,
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

    # Update the split files with the new augmented dataset
    create_split_augmented(file_path, f'{file_name}_augmented.pkl')
    print(f'{file_path} was correctly augmented')

def delete_files():
    emg_folder = 'emg/'
    action_folder = 'action-net/'
    
    for filename in os.listdir(emg_folder):
        if os.path.isfile(os.path.join(emg_folder, filename)):
            if 'specto' in filename.lower():
                os.remove(os.path.join(emg_folder, filename))

    return

    for filename in os.listdir(action_folder):
        if os.path.isfile(os.path.join(action_folder, filename)):
            if 'augmented' in filename.lower():
                os.remove(os.path.join(action_folder, filename))

    for filename in os.listdir(emg_folder):
        if os.path.isfile(os.path.join(emg_folder, filename)):
            if 'augmented' in filename.lower():
                os.remove(os.path.join(emg_folder, filename))

def augment_dataset():
    emg_folder = 'emg/'
    
    partitions = os.listdir(emg_folder)
    for p in partitions:
        augment_partition(os.path.join(emg_folder, p))

# def update_split_files():
#     test = pd.DataFrame(pd.read_pickle('action-net/ActionNet_test_augmented.pkl'))
#     train = pd.DataFrame(pd.read_pickle('action-net/ActionNet_train_augmented.pkl'))

#     if 'file' in train.columns:
#         train = train.rename(columns={'file':'emg_file'})

#     if 'file' in test.columns:
#         test = test.rename(columns={'file':'emg_file'})

#     for i, row in train.iterrows():
#         filename = row['emg_file']
#         if 'preproc' not in filename:
#             file_name, file_extension = os.path.splitext(filename)
#             train.at[i, 'emg_file'] = os.path.join(filename, '_preproc.pkl')

#     for i, row in test.iterrows():
#         filename = row['emg_file']
#         if 'preproc' not in filename:
#             file_name, file_extension = os.path.splitext(filename)
#             test.at[i, 'emg_file'] = os.path.join(filename, '_preproc.pkl')

#     if 'rgb_file' not in train.columns:
#         train.insert(2, 'rgb_file', [train.loc[i, 'emg_file'].replace('_preproc', '_rgb') for i in range(len(train))], allow_duplicates=True)

#     if 'rgb_file' not in test.columns:
#         test.insert(2, 'rgb_file', [test.loc[i, 'emg_file'].replace('_preproc', '_rgb') for i in range(len(test))], allow_duplicates=True)

#     if 'spectograms_file' not in train.columns:
#         train.insert(2, 'spectograms_file', [train.loc[i, 'emg_file'].replace('_preproc', '_specto') for i in range(len(train))], allow_duplicates=True)

#     if 'spectograms_file' not in test.columns:
#         test.insert(2, 'spectograms_file', [test.loc[i, 'emg_file'].replace('_preproc', '_specto') for i in range(len(test))], allow_duplicates=True)

#     return

def emg2rgb():
    emg_folder = 'emg/'
    partitions = os.listdir(emg_folder)

    for filename in partitions:
        if os.path.isfile(os.path.join(emg_folder, filename)) and 'augmented' in filename.lower():
            data = pd.DataFrame(pd.read_pickle(os.path.join(emg_folder, filename)))
            new_data = []

            for i, _ in data.iterrows():
                new_row = {
                    'uid': i,
                    'participant_id': 'P04',
                    'video_id': 'P04_01',
                    'narration': data.loc[i, 'description'],
                    'start_timestamp': data.loc[i, 'start'],
                    'stop_timestamp': data.loc[i, 'stop'],
                    'start_frame': data.loc[i, 'start'] * 29.67,
                    'stop_frame': data.loc[i, 'stop'] * 29.67,
                    'verb': data.loc[i, 'description'],
                    'verb_class': emg_descriptions_to_labels.index(data.loc[i, 'description']),
                }
                new_data.append(new_row)

            # Convert the list of dictionaries to a DataFrame
            new_data = pd.DataFrame(new_data)

            # Save the DataFrame to a pickle file
            file_name, file_extension = os.path.splitext(filename)
            new_data.to_pickle(f'{emg_folder}/{file_name.split("_emg")[0]}_rgb.pkl')
            print(f'{filename} succesfully produced the rgb counterpart')

def merge_pickles():
    agents = {} 
    emg_folder = 'emg/'
    action_folder = 'action-net/'

    for filename in os.listdir(emg_folder):
        if os.path.isfile(os.path.join(emg_folder, filename)):
            agent = filename[:5]
            if agent not in agents.keys():
                agents[agent] = {'emg_file': 'none', 'rgb_file': 'none', 'spectograms_file': 'none'}
            if 'emg' in filename.lower():
                agents[agent]['emg_file'] = filename
            elif 'rgb' in filename.lower():
                agents[agent]['rgb_file'] = filename
            elif 'specto' in filename.lower():
                agents[agent]['spectograms_file'] = filename
            else:
                continue

    for agent, files in agents.items():
        emg = pd.DataFrame(pd.read_pickle(os.path.join(emg_folder, files['emg_file'])))
        rgb = pd.DataFrame(pd.read_pickle(os.path.join(emg_folder, files['rgb_file'])))
        specto = pd.DataFrame(pd.read_pickle(os.path.join(emg_folder, files['spectograms_file'])))

        # remove t0 from timestamps and frame nubers T0
        # >>> min(emg['start'])
        # 1654639217.5249205 ====> inizio video = frame_t0=49093145568
        # 1654639888.5249205 ====> calibration start = 11m11s
        # 1654640077,704248 ====> prima azione = 14m20s
        def remove_t0_time(value: float):
            if type(value)==list:
                return [t - 1654639217.5249205 for t in value]
            else:
                return value - 1654639217.5249205

        def remove_t0_frame(value: int):
            return value - 49093145568

        emg['start'] = emg['start'].map(remove_t0_time)
        emg['stop'] = emg['stop'].map(remove_t0_time)
        emg['myo_left_timestamps'] = emg['myo_left_timestamps'].map(remove_t0_time)
        emg['myo_right_timestamps'] = emg['myo_right_timestamps'].map(remove_t0_time)
        rgb['start_frame'] = rgb['start_frame'].map(remove_t0_frame)
        rgb['stop_frame'] = rgb['stop_frame'].map(remove_t0_frame)

        # keep only necessary columns
        emg = emg.loc[:, [
            'start',
            'stop',
            'myo_left_timestamps', 
            'myo_left_readings', 
            'myo_right_timestamps', 
            'myo_right_readings', 
            'description', 
            ]]
        rgb = rgb.loc[:, [
            'start_frame', 
            'stop_frame',
            'verb_class'
            ]]
        specto = specto.loc[:, [
            'file',
            ]]

        # rename columns
        emg = emg.rename(columns={'start': 'start_timestamp', 'stop': 'stop_timestamp'})
        rgb = rgb.rename(columns={'verb_class': 'label'})
        specto = specto.rename(columns={'file': 'specto_file'})

        # join columns
        final = pd.merge(emg, rgb, left_index=True, right_index=True)
        final = pd.merge(final, specto, left_index=True, right_index=True)

        final.to_pickle(f'{emg_folder}/{agent}_augmented.pkl')

        # modify split files, so they will point to the new files
        def map_new_file(value: str):
            file_name, file_extension = os.path.splitext(value)
            return file_name[:5] + '_augmented.pkl'
            
        test = pd.DataFrame(pd.read_pickle(os.path.join(action_folder, 'ActionNet_test_augmented.pkl')))
        train = pd.DataFrame(pd.read_pickle(os.path.join(action_folder, 'ActionNet_train_augmented.pkl')))
        test['file'] = test['file'].map(map_new_file)
        train['file'] = train['file'].map(map_new_file)
        test.to_pickle(f'{action_folder}/ActionNet_test_augmented.pkl')
        train.to_pickle(f'{action_folder}/ActionNet_train_augmented.pkl')

    for filename in os.listdir(emg_folder):
        if os.path.isfile(os.path.join(emg_folder, filename)) and ('rgb' in filename.lower() or 'emg' in filename.lower() or 'specto' in filename.lower()):
            os.remove(os.path.join(emg_folder, filename))
            
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

    tmp_lefts, tmp_rights = data["myo_left_readings"], data["myo_right_readings"]
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
                aus = np.vstack((aus, preprocessed[start + l]))
            
            #SUM EACH CHANNEL FOR EACH PERIOD
            data.at[i, side_name] = aus
            
            start += period_length
    
    put_back_into_dataframe("myo_left_readings", filtered_data_left, length_periods_l)
    put_back_into_dataframe("myo_right_readings", filtered_data_right, length_periods_r)
    
    return data

def final_save_spectrogram(specgram_l, specgram_r, name, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(len(specgram_l)+len(specgram_r), 1, figsize=(16, 8))

    axs[0].set_title(title or "Spectrogram (db)")

    both_specs = [*specgram_l, *specgram_r]

    for i, spec in enumerate(both_specs):
        im = axs[i].imshow(librosa.power_to_db(both_specs[i]), origin="lower", aspect="auto")
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
    
    axs[i].set_xlabel("Frame number")
    axs[i].get_xaxis().set_visible(True)
            
    plt.savefig(f"../spectograms/{name}")
    plt.close()

def save_spectograms(skipSectrograms = 0):
    backup_spectrograms = {'Get/replace items from refrigerator/cabinets/drawers': 'S00_2_0.png', 'Peel a cucumber': 'S00_2_15.png', 'Slice a cucumber': 'S00_2_43.png', 'Peel a potato': 'S00_2_72.png', 'Slice a potato': 'S00_2_104.png', 'Slice bread': 'S00_2_134.png', 'Spread almond butter on a bread slice': 'S00_2_165.png', 'Spread jelly on a bread slice': 'S00_2_180.png', 'Open/close a jar of almond butter': 'S00_2_189.png', 'Pour water from a pitcher into a glass': 'S00_2_201.png', 'Clean a plate with a sponge': 'S00_2_224.png', 'Clean a plate with a towel': 'S00_2_236.png', 'Clean a pan with a sponge': 'S00_2_243.png', 'Clean a pan with a towel': 'S00_2_251.png', 'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 'S00_2_260.png', 'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 'S00_2_282.png', 'Stack on table: 3 each large/small plates, bowls': 'S00_2_304.png', 'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 'S00_2_315.png', 'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 'S00_2_350.png', 'Clear cutting board': 'S02_2_48.png'}
    
    n_fft = 32
    win_length = None
    hop_length = 4

    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        normalized=True
    )

    def compute_spectrogram(signal):
        freq_signal = [spectrogram(signal[:, i]) for i in range(8)]
        return freq_signal
        
    files_to_read = [s for s in os.listdir('emg/') if "augmented" in s and 'rgb' not in s]
    
    L = 0
    
    for i, f in enumerate(files_to_read):
        cur_values = []
        agent = f[0:5]
        print(f"{i+1}/{len(files_to_read)}: {f}")
        emg_annotations = pd.read_pickle(f"emg/{f}")
        for sample_no in range(len(emg_annotations)):
            signal_l = torch.from_numpy(emg_annotations.iloc[sample_no].myo_left_readings).float()
            signal_r = torch.from_numpy(emg_annotations.iloc[sample_no].myo_right_readings).float()
            label = emg_annotations.iloc[sample_no].description
            file_name_prefix = f.split("_augmented")[0]
            name = f"{file_name_prefix}_{sample_no}.png"
            try:
                freq_signal_l = compute_spectrogram(signal_l)
                freq_signal_r = compute_spectrogram(signal_r)
            except RuntimeError:
                new_row_data = [backup_spectrograms[label], label]
                cur_values.append(new_row_data)
            else:
                # uncomment this line if you want to save the spectogram in the folder of spectograms
                if not skipSectrograms:
                    final_save_spectrogram(freq_signal_l, freq_signal_r, name, title=label)
                new_row_data = [name, label]
                cur_values.append(new_row_data)
        
        cur_df = pd.DataFrame(cur_values, columns=['file','description'])
        cur_df.to_pickle(f"emg/{agent}_augmented_specto.pkl")
        
        L+=len(cur_df)
            
    print("FINAL: " + str(L))
        
def pre_process_emg():
    emg_folder = 'emg/'

    for filename in os.listdir(emg_folder):
        if os.path.isfile(os.path.join(emg_folder, filename)) and 'augmented' in filename.lower():
            data = emg_adjust_features(os.path.join(emg_folder, filename))
            os.remove(os.path.join(emg_folder, filename))
            data.to_pickle(f'{emg_folder}/{filename.split(".pkl")[0]}_emg.pkl')
            print(f'{emg_folder}/{filename} was correctly preprocessed')

def pipeline():
    delete_files()
    #augment_dataset()
    #pre_process_emg()
    #emg2rgb()
    save_spectograms(skipSectrograms=1)
    merge_pickles()


if __name__ == '__main__':
    pipeline()
