import pandas as pd
from scipy import signal, interpolate
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import math

import imageio
import cv2

import torch
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


def chunk_timestamps_and_readings(left_timestamps, left_readings, right_timestamps, right_readings, time_interval:int=5):
    # Initialize empty lists to store the chunks and readings
    left_chunks = [[]]
    right_chunks = [[]]
    left_reading_chunks = [[]]
    right_reading_chunks = [[]]

    # Initialize variables to track the current chunk start and end
    left_chunk_start = left_timestamps[0]
    left_chunk_end = left_chunk_start + time_interval
    right_chunk_start = right_timestamps[0]
    right_chunk_end = right_chunk_start + time_interval

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
            left_chunk_end = left_chunk_start + time_interval

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
            right_chunk_end = right_chunk_start + time_interval

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
    for i, row in data.iterrows():
        old_index = row['old_index']
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

        for new_index in mapping_new_index[old_index] if old_index in mapping_new_index.keys() else []:
            # handle old descriptions
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
        for new_index in mapping_new_index[old_index] if old_index in mapping_new_index.keys() else []:
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

def padding(sample, size):
    left_readings = sample['myo_left_readings']
    right_readings = sample['myo_right_readings']

    readings = {'myo_left_readings': left_readings, 'myo_right_readings': right_readings}

    for key, value in readings.items():
        original_length = len(value)
        diff = size - original_length

        zeros_left = diff // 2
        zeros_right = diff - zeros_left

        average_value = sum(value)/len(value)

        # Pad the list with zeros on both sides
        padded_list = [average_value] * zeros_left + value + [average_value] * zeros_right
        sample[key] = padded_list

    return sample

def augment_partition(file_path: str):
    data = pd.DataFrame(pd.read_pickle(file_path))
    new_data = []

    for i, row in data.iterrows():
        if row['description'] == 'calibration':
            continue
        left_timestamps_chunks, right_timestamps_chunks, left_readings_chunks, right_readings_chunks = chunk_timestamps_and_readings(row['myo_left_timestamps'], row['myo_left_readings'], row['myo_right_timestamps'], row['myo_right_readings'])

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
                'stop': max(left_timestamps_chunks[c][-1], right_timestamps_chunks[c][-1]),
                'myo_left_timestamps': left_timestamps_chunks[c],
                'myo_left_readings': left_readings_chunks[c],
                'myo_right_timestamps': right_timestamps_chunks[c],
                'myo_right_readings': right_readings_chunks[c]
                }
            new_data.append(new_row)

    max_length_sample = max([max(len(sample['myo_left_readings']), len(sample['myo_right_readings'])) for sample in new_data])
    threshold = max_length_sample*3/4
    new_data = list(filter(lambda sample: min(len(sample['myo_left_readings']), len(sample['myo_right_readings'])) > threshold, new_data))
    new_data = list(map(lambda sample: padding(sample, max_length_sample), new_data))

    # Convert the list of dictionaries to a DataFrame
    new_data = pd.DataFrame(new_data)

    # Save the DataFrame to a pickle file
    file_name, file_extension = os.path.splitext(file_path)
    file_name = file_name.replace('_resample', '')
    new_data.to_pickle(f'{file_name}_augmented.pkl')

    # Update the split files with the new augmented dataset
    file_path = file_path.replace('_resample', '')
    create_split_augmented(file_path, f'{file_name}_augmented.pkl')
    print(f'{file_path} was correctly augmented')

def delete_files():
    emg_folder = 'emg/'
    action_folder = 'action-net/'

    for filename in os.listdir(action_folder):
        if os.path.isfile(os.path.join(action_folder, filename)):
            if 'augmented' in filename.lower() or 'resample' in filename.lower():
                os.remove(os.path.join(action_folder, filename))

    for filename in os.listdir(emg_folder):
        if os.path.isfile(os.path.join(emg_folder, filename)):
            if 'augmented' in filename.lower() or 'resample' in filename.lower():
                os.remove(os.path.join(emg_folder, filename))

def pad_partitions():
    emg_folder = 'emg/'

    partitions = os.listdir(emg_folder)
    files = {}
    for p in partitions:
        if 'augmented' in p:
            files[p] = pd.DataFrame(pd.read_pickle(os.path.join(emg_folder, p)))

    fixed_length = max([len(f.loc[0, 'myo_left_readings']) for f in files.values()])

    for filename, dataframe in files.items():
        for i in range(len(dataframe)):
            dataframe.iloc[i, :] = padding(dataframe.iloc[i, :], fixed_length)

        dataframe.to_pickle(os.path.join(emg_folder, filename))

def augment_dataset():
    emg_folder = 'emg/'

    partitions = os.listdir(emg_folder)
    for p in partitions:
        if 'resample' in p:
            augment_partition(os.path.join(emg_folder, p))

    pad_partitions()

def emg2rgb():
    emg_folder = 'emg/'
    partitions = os.listdir(emg_folder)
    frame_rate = 29.67

    for filename in partitions:
        if os.path.isfile(os.path.join(emg_folder, filename)) and 'augmented' in filename.lower():
            data = pd.DataFrame(pd.read_pickle(os.path.join(emg_folder, filename)))
            new_data = []


            for i, row in data.iterrows():
                new_row = {
                    'start_timestamp': row['start'],
                    'stop_timestamp': row['stop'],
                    'start_frame': row['start'] * frame_rate,
                    'stop_frame': row['stop'] * frame_rate,
                    'verb': row['description'],
                    'verb_class': emg_descriptions_to_labels.index(row['description']),
                }
                new_data.append(new_row)

            # Convert the list of dictionaries to a DataFrame
            new_data = pd.DataFrame(new_data)

            # Save the DataFrame to a pickle file
            file_name, file_extension = os.path.splitext(filename)
            new_data.to_pickle(f'{emg_folder}/{file_name.split("_emg")[0]}_rgb.pkl')
            print(f'{filename} succesfully produced the rgb counterpart')

def remove_t0_time(value: float, t0: float, frame_rate: float):
    if type(value)==list:
        return [t - t0 for t in value]
    else:
        return value - t0

def remove_t0_frame(value: float, t0: float, frame_rate: float):
    # remove t0. add 1 because the first frame is 0000001 and not 0000000
    return int(value - (t0*frame_rate) + 1)

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
        # 1655239114.183343 ====> calibration start
        frame_rate = 29.62
        t0 = min(emg['start'])

        emg['start'] = emg['start'].map(lambda x: remove_t0_time(x, t0=t0, frame_rate=29.67))
        emg['stop'] = emg['stop'].map(lambda x: remove_t0_time(x, t0=t0, frame_rate=29.67))
        emg['myo_left_timestamps'] = emg['myo_left_timestamps'].map(lambda x: remove_t0_time(x, t0=t0, frame_rate=29.67))
        emg['myo_right_timestamps'] = emg['myo_right_timestamps'].map(lambda x: remove_t0_time(x, t0=t0, frame_rate=29.67))
        rgb['start_frame'] = rgb['start_frame'].map(lambda x: remove_t0_frame(x, t0=t0, frame_rate=29.67))
        rgb['stop_frame'] = rgb['stop_frame'].map(lambda x: remove_t0_frame(x, t0=t0, frame_rate=29.67))

        # keep only necessary columns
        emg = emg.loc[:, [
            'myo_left_readings',
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
        if os.path.isfile(os.path.join(emg_folder, filename)) and ('rgb' in filename.lower() or 'emg' in filename.lower() or 'specto' in filename.lower() or or 'resample' in filename.lower()):
            os.remove(os.path.join(emg_folder, filename))

def design_lowpass_filter(cutoff_freq, sample_rate, filter_order=1):
    nyquist_freq = sample_rate * 0.5  # Nyquist frequency
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(filter_order, normalized_cutoff, btype='lowpass')  # Butterworth filter
    return b, a

def scale_and_normalize(data):
    data = np.array(data)
    data = (data - data.min()) / (data.max() - data.min()) * 2 - 1  
    data = (data - data.mean()) / data.std()
    return data

def filter_signal(data, b, a):
    return signal.filtfilt(b, a, data)

def emg_adjust_features(file_path: str, *, cut_frequency: float = 5.0, filter_order: int = 2):
    data = pd.DataFrame(pd.read_pickle(file_path))
    
    fs = 15                    # sampling frequency
    filter_b, filter_a = design_lowpass_filter(cut_frequency, fs, filter_order)
    
    for side in ['myo_left_readings', 'myo_right_readings']:
        np_side_data = np.empty((0,8))
        lengths = []
        for sample in data[side]:
            np_sample = np.array(sample)
            lengths.append(np_sample.shape[0])
            np_side_data = np.vstack((np_side_data, np_sample))
        
        for j in range(8):
            np_side_data[:,j] = filter_signal(np_side_data[:,j], filter_b, filter_a)
            np_side_data[:,j] = scale_and_normalize(np_side_data[:,j])
            
        start = 0
        for i, l in enumerate(lengths):
            data.iat[i, data.columns.get_loc(side)] = np_side_data[start:start+l, :].tolist()
            start+=l

    return data

def final_save_spectrogram(specgram_l, specgram_r, name, resize_factor=.25):
    fig, axs = plt.subplots(len(specgram_l)+len(specgram_r), 1, figsize=(16, 8))

    # Remove the title (axs[0].set_title(...) is commented out)
    # axs[0].set_title(title or "Spectrogram (db)")

    both_specs = [*specgram_l, *specgram_r]

    for i, spec in enumerate(both_specs):
        im = axs[i].imshow(librosa.power_to_db(both_specs[i]), origin="lower", aspect="auto")
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)

    # Move x-axis label to the last subplot and show the x-axis
    axs[-1].set_xlabel("Frame number")

    # Adjust the layout to remove whitespace
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    fig = plt.gcf()

    # Convert the figure to a numpy array
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Resize
    image_from_plot = cv2.resize(image_from_plot, dsize=None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

    # Save as an image (you can choose the format based on your needs)
    imageio.imwrite(f"../spectograms/{name}", image_from_plot)
    plt.close()

def save_spectograms(skipSectrograms=False):
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

    files_to_read = [s for s in os.listdir('emg/') if "augmented" in s and 'rgb' not in s and 'S09_2' in s]

    print("Saving spectrograms...")

    for i, f in enumerate(files_to_read):
        cur_values = []
        agent = f[0:5]
        emg_annotations = pd.read_pickle(f"emg/{f}")
        for sample_no in range(len(emg_annotations)):
            signal_l = torch.Tensor(emg_annotations.iloc[sample_no].myo_left_readings)
            signal_r = torch.Tensor(emg_annotations.iloc[sample_no].myo_right_readings)
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
                    final_save_spectrogram(freq_signal_l, freq_signal_r, name)
                new_row_data = [name, label]
                cur_values.append(new_row_data)
        
        cur_df = pd.DataFrame(cur_values, columns=['file','description'])
        cur_df.to_pickle(f"emg/{agent}_augmented_specto.pkl")

        print(f"emg/{agent}_augmented_specto.pkl correctly generated")

def pre_process_emg():
    emg_folder = 'emg/'

    for filename in os.listdir(emg_folder):
        if os.path.isfile(os.path.join(emg_folder, filename)) and 'augmented' in filename.lower():
            data = emg_adjust_features(os.path.join(emg_folder, filename))
            os.remove(os.path.join(emg_folder, filename))
            data.to_pickle(f'{emg_folder}/{filename.split(".pkl")[0]}_emg.pkl')
            print(f'{emg_folder}{filename} was correctly preprocessed')

def balance_train_test_split(threshold_proportion=0.05):
    split_train = pd.DataFrame(pd.read_pickle('action-net/ActionNet_train_augmented.pkl'))
    split_test = pd.DataFrame(pd.read_pickle('action-net/ActionNet_test_augmented.pkl'))
    samples = {}
    dataset_train = []
    dataset_test = []
    n_samples_x_class = {'train': np.zeros(20), 'test': np.zeros(20)}

    for filename in os.listdir('emg/'):
        if 'augmented' in filename:
            agent = filename[:5]
            samples_agent = pd.DataFrame(pd.read_pickle(os.path.join('emg/', filename)))
            samples[agent] = samples_agent

    for i, row in split_train.iterrows():
        index = row['index']
        filename = row['file']
        agent = filename[:5]
        sample = samples[agent].loc[index, :]
        sample = sample.copy()
        sample['from_file'] = filename
        dataset_train.append(sample)

    dataset_train = pd.DataFrame(dataset_train)

    num_classes = 20
    for c in range(num_classes):
        tot_samples_c = len(dataset_train[dataset_train['label'] == c])
        n_samples_x_class['train'][c] = tot_samples_c

    for i, row in split_test.iterrows():
        index = row['index']
        filename = row['file']
        agent = filename[:5]
        sample = samples[agent].loc[index, :]
        sample = sample.copy()
        sample['from_file'] = filename
        dataset_test.append(sample)

    dataset_test = pd.DataFrame(dataset_test)

    for c in range(num_classes):
        tot_samples_c = len(dataset_test[dataset_test['label'] == c])
        n_samples_x_class['test'][c] = tot_samples_c

    # print(f'proportions before the balance:')    
    # for c in range(num_classes):
    #     p = n_samples_x_class['test'][c] / n_samples_x_class['train'][c]
    #     print(f'class {c}: {p}')

    for c in range(num_classes):
        candidates_to_move = dataset_train[dataset_train['label'] == c]       
        if n_samples_x_class['test'][c] / n_samples_x_class['train'][c] <= threshold_proportion:
            print(f'balancing class {c}...')
            n_to_move = math.ceil(threshold_proportion*n_samples_x_class['train'][c] - n_samples_x_class['test'][c])
            to_move = candidates_to_move.head(n_to_move)

            print(f'moving {n_to_move} samples from tarin to test:')
            print(to_move)

            n_samples_x_class['train'][c] -= n_to_move
            n_samples_x_class['test'][c] += n_to_move

            for i, row in to_move.iterrows():
                row_to_move = split_train[(split_train['file']==row['from_file']) & (split_train['index']==i)]
                split_train = split_train.drop(row_to_move.index)
                split_test = pd.concat([split_test, row_to_move], ignore_index=True)

            split_train.to_pickle(f"action-net/ActionNet_train_augmented.pkl")
            split_test.to_pickle(f"action-net/ActionNet_test_augmented.pkl")

    # print(f'proportions after the balance:')    
    # for c in range(num_classes):
    #     p = n_samples_x_class['test'][c] / n_samples_x_class['train'][c]
    #     print(f'class {c}: {p}')

    print('split files were correctly balanced')

def resample(sampling_rate:float=15.):
    emg_folder = 'emg/'
    partitions = os.listdir(emg_folder)

    files = {}
    for p in partitions:
        files[p] = pd.DataFrame(pd.read_pickle(os.path.join(emg_folder, p)))

    sampling_interval = 1/sampling_rate

    for filename, dataframe in files.items():
        dataframe = dataframe[dataframe['description'] != 'calibration']
        timestamps_sx = np.concatenate(dataframe['myo_left_timestamps'].values, axis=-1)
        timestamps_dx = np.concatenate(dataframe['myo_right_timestamps'].values, axis=-1)
        readings_sx = np.concatenate(dataframe['myo_left_readings'].values, axis=0).transpose(1, 0)
        readings_dx = np.concatenate(dataframe['myo_right_readings'].values, axis=0).transpose(1, 0)

        fn_interpolate_sx = [interpolate.interp1d(
            timestamps_sx,       
            readings_sx[ix],         
            axis=0,           
            kind='linear',    
            fill_value='extrapolate' 
        ) for ix in range(8)]
        fn_interpolate_dx = [interpolate.interp1d(
            timestamps_dx,       
            readings_dx[ix],         
            axis=0,           
            kind='linear',    
            fill_value='extrapolate' 
        ) for ix in range(8)]

        for i, row in dataframe.iterrows():
            new_timestamps_sx = np.arange(row['myo_left_timestamps'][0], row['myo_left_timestamps'][-1], sampling_interval)
            new_timestamps_dx = np.arange(row['myo_right_timestamps'][0], row['myo_right_timestamps'][-1], sampling_interval)
            dataframe.at[i, 'myo_left_timestamps'] = new_timestamps_sx
            dataframe.at[i, 'myo_right_timestamps'] = new_timestamps_dx
            dataframe.at[i, 'myo_left_readings'] = np.array([fn_interpolate_sx[ix](new_timestamps_sx) for ix in range(8)]).transpose(1, 0)
            dataframe.at[i, 'myo_right_readings'] = np.array([fn_interpolate_dx[ix](new_timestamps_dx) for ix in range(8)]).transpose(1, 0)

        file_name, file_extension = os.path.splitext(filename)
        dataframe.to_pickle(os.path.join(emg_folder, file_name + "_resample.pkl"))

        print(f'{filename} was correctly resampled')

def pipeline():
    # delete_files()
    # resample()
    # augment_dataset()
    # pre_process_emg()
    # emg2rgb()
    save_spectograms(skipSectrograms=False)
    merge_pickles()
    balance_train_test_split()

if __name__ == '__main__':
    pipeline()
