from scipy.signal import butter, lfilter, filtfilt, resample_poly, resample
from scipy import signal, interpolate
from typing import Tuple
from copy import deepcopy
import pandas as pd
import numpy as np
import torchaudio.transforms as T
import torch
import os
import math
import cv2
import librosa
import pickle
import h5py
from torchvision import transforms

class ProcessEmgDataset():
    def __init__(self, emg_checkpoint_folder:str=None, split_checkpoint_folder:str=None, rgb_folder:str=None, specto_folder:str=None):
        self.FOLDERS = {
            'data': 'emg',
            'split': 'action-net',
        }

        self.DESCRIPTIONS_TO_LABELS = [
            'Get/replace items from refrigerator/cabinets/drawers',
            'Peel a cucumber',
            'Clear cutting board',
            'Slice a cucumber',
            'Peel a potato',
            'Slice a potato',
            'Slice bread',
            'Spread almond butter on a bread slice',
            'Spread jelly on a bread slice',
            'Open/close a jar of almond butter',
            'Pour water from a pitcher into a glass',
            'Clean a plate with a sponge',
            'Clean a plate with a towel',
            'Clean a pan with a sponge',
            'Clean a pan with a towel',
            'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
            'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
            'Stack on table: 3 each large/small plates, bowls',
            'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
            'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
        ]

        self.DESCRIPTIONS_CONVERSION_DICT = {
            'Get items from refrigerator/cabinets/drawers'         :       'Get/replace items from refrigerator/cabinets/drawers',
            'Open a jar of almond butter'                          :       'Open/close a jar of almond butter'
        }

        self.FRAME_RATE = 29.67
        self.split_files = {
            'train': 'ActionNet_train.pkl',
            'test': 'ActionNet_test.pkl',
        }
        self.rgb_folder = rgb_folder
        self.specto_folder = specto_folder
        self.current_step = 1
        self.current_emg_folder = emg_checkpoint_folder if emg_checkpoint_folder else 'base'
        self.current_split_folder = split_checkpoint_folder if split_checkpoint_folder else 'base' 

    def __get_agents__(self) -> list[str]:
        agents = list()
        for filename in os.listdir(os.path.join(self.FOLDERS['data'], 'base')):
            agents.append(filename[:5])
        return agents

    def __pad_readings__(self, data, size:int, type_padding:str='zeros'):
        new_data = deepcopy(data)
        original_length = len(new_data)
        diff = size - original_length

        if diff == 0:
            return new_data

        left_padding_lenght = diff // 2
        right_padding_lenght = diff - left_padding_lenght
        
        if type_padding == "right_only_0":
            new_data = np.pad(new_data, ((0, diff), (0, 0)), mode='constant')
        else:
            # Pad the list with zeros on both sides
            if left_padding_lenght > 0:
                if type_padding == 'noise':
                    average_value = np.average(new_data, axis=0)
                    pad_value = np.array(average_value).reshape(1, 8)
                    left_padding = pad_value + np.random.normal(0, 1, (left_padding_lenght, 8))
                elif type_padding == 'zeros':
                    left_padding = np.zeros(8).reshape(1, 8).repeat(left_padding_lenght, axis=0)
                else:
                    average_value = np.average(new_data, axis=0)
                    pad_value = np.array(average_value).reshape(1, 8)
                    left_padding = pad_value.repeat(left_padding_lenght).reshape(-1, 8)
                new_data = np.concatenate((left_padding, new_data), axis=0)
            if right_padding_lenght > 0:
                if type_padding == 'noise':
                    average_value = np.average(new_data, axis=0)
                    pad_value = np.array(average_value).reshape(1, 8)
                    right_padding = pad_value + np.random.normal(0, 1, (right_padding_lenght, 8))
                elif type_padding == 'zeros':
                    right_padding = np.zeros(8).reshape(1, 8).repeat(right_padding_lenght, axis=0)
                else:
                    average_value = np.average(new_data, axis=0)
                    pad_value = np.array(average_value).reshape(1, 8)
                    right_padding = pad_value.repeat(right_padding_lenght, axis=0)
                new_data = np.concatenate((new_data, right_padding), axis=0)

        return new_data

    def __multiply_actions__(self, left_timestamps:list, left_readings:list, right_timestamps:list, right_readings:list, time_interval:int=10, num_chunks:int=20, sampling_rate:float=5, type_padding:str='zeros') -> Tuple[list]:
        # Initialize empty lists to store the chunks and readings
        left_timestamps_chunks = list()
        right_timestamps_chunks = list()
        left_readings_chunks = list()
        right_readings_chunks = list()

        timestamps_per_action = int(sampling_rate*time_interval)

        if min(len(left_timestamps), len(right_timestamps)) < timestamps_per_action:
            left_readings = self.__pad_readings__(left_readings, timestamps_per_action, type_padding)
            right_readings = self.__pad_readings__(right_readings, timestamps_per_action, type_padding)
            return [left_timestamps], [right_timestamps], [left_readings], [right_readings]
        else:
            starting_points = np.linspace(left_timestamps[0], left_timestamps[-1] - time_interval, num=num_chunks)
            for sp in starting_points:
                start_idx = np.where(left_timestamps >= sp)[0][0]
                end_idx = start_idx + timestamps_per_action

                if end_idx >= len(left_timestamps):
                    break

                left_readings_chunks.append(self.__pad_readings__(left_readings[start_idx:end_idx], timestamps_per_action, type_padding))
                right_readings_chunks.append(self.__pad_readings__(right_readings[start_idx:end_idx], timestamps_per_action, type_padding))
                left_timestamps_chunks.append(left_timestamps[start_idx:end_idx])
                right_timestamps_chunks.append(right_timestamps[start_idx:end_idx])    

        return left_timestamps_chunks, right_timestamps_chunks, left_readings_chunks, right_readings_chunks

    def __remove_t0_time__(self, value: float, t0: float) -> float:
        if type(value)==list:
            return [t - t0 for t in value]
        else:
            return value - t0

    def __remove_t0_frame__(self, value: float, t0: float) -> int:
        # remove t0. add 1 because the first frame is 0000001 and not 0000000
        return int(value - (t0*self.FRAME_RATE) + 1)

    def __update_split_files__(self) -> None:
        train = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['split'], self.current_split_folder, self.split_files['train'])))
        test = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['split'], self.current_split_folder, self.split_files['test'])))
        
        for filename in os.listdir(os.path.join(self.FOLDERS['data'], self.current_emg_folder)):
            data = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['data'], self.current_emg_folder, filename)))

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
            old_test_rows = test[test['file'] == filename]
            new_test_rows = []
            new_data = []

            for i, row in old_test_rows.iterrows():
                old_index = row['index']

                for new_index in mapping_new_index[old_index] if old_index in mapping_new_index.keys() else []:
                    # handle old descriptions
                    new_row = {
                        'index': int(new_index),
                        'file': filename,
                        'description': row['description'] if row['description'] not in self.DESCRIPTIONS_CONVERSION_DICT.keys() else self.DESCRIPTIONS_CONVERSION_DICT[row['description']],
                        'labels': row['labels']
                    }
                    new_data.append(new_row)

            new_test_rows = test[test['file'] != filename]
            new_test_rows = pd.concat([new_test_rows, pd.DataFrame(new_data)])

            # TRAIN: creating the new rows of the new file split that refer to the augmented dataset
            old_train_rows = train[train['file'] == filename]
            new_train_rows = []
            new_data = []

            for i, row in old_train_rows.iterrows():
                old_index = row['index']
                for new_index in mapping_new_index[old_index] if old_index in mapping_new_index.keys() else []:
                    # handle old descriptions
                    new_row = {
                        'index': int(new_index),
                        'file': filename,
                        'description': row['description'] if row['description'] not in self.DESCRIPTIONS_CONVERSION_DICT.keys() else self.DESCRIPTIONS_CONVERSION_DICT[row['description']],
                        'labels': row['labels']
                    }
                    new_data.append(new_row)

            new_train_rows = train[train['file'] != filename]
            new_train_rows = pd.concat([new_train_rows, pd.DataFrame(new_data)])

            new_train_rows.set_index('index', inplace=True, drop=False)
            new_test_rows.set_index('index', inplace=True, drop=False)
            
            train = new_train_rows
            test = new_test_rows

        # Save the DataFrame to a pickle file
        self.current_split_folder = f'step{self.current_step}-augment'
        if not os.path.exists(os.path.join(self.FOLDERS['split'], self.current_split_folder)):
            os.makedirs(os.path.join(self.FOLDERS['split'], self.current_split_folder))
        train.to_pickle(os.path.join(self.FOLDERS['split'], self.current_split_folder, self.split_files['train']))
        test.to_pickle(os.path.join(self.FOLDERS['split'], self.current_split_folder, self.split_files['test']))

    def __low_pass_filter__(self, data, cut_frequency:float=5., filter_order:int=2, data_target="channel") -> pd.DataFrame:        
        if data_target == "sample":
            for i, sample in data.iterrows():
                for side in ['myo_left_readings', 'myo_right_readings']:
                    np_sample = np.array(sample[side])
                    
                    t = sample[f"{side.split('_')[0]}_{side.split('_')[1]}_timestamps"]
                    fs = (t.size - 1) / (t[-1] - t[0])
                    
                    nyquist_freq = fs * 0.5  # Nyquist frequency
                    normalized_cutoff = cut_frequency / nyquist_freq
                    b, a = butter(filter_order, normalized_cutoff, btype='lowpass', analog=False)  # Butterworth filter
                    
                    np_sample = lfilter(b, a, np_sample, axis=0)
                    
                    data.at[i, side] = np_sample

        elif data_target == 'channel':
            for side in ['myo_left_readings', 'myo_right_readings']:
                np_side_data = np.empty((0,8))
                lengths = []
                for sample in data[side]:
                    np_sample = np.array(sample)
                    lengths.append(np_sample.shape[0])
                    np_side_data = np.vstack((np_side_data, np_sample))
                
                for j in range(8):
                    np_side_data[:,j] = signal.filtfilt(b, a, np_side_data[:,j])
                    
                start = 0
                for i, l in enumerate(lengths):
                        data.iat[i, data.columns.get_loc(side)] = np_side_data[start:start+l, :].tolist()
                        start+=l
        return data

    def __normalize__(self, data, data_target:str='channel_global') -> pd.DataFrame: 
        stats = dict
        with open('stats.pkl', 'rb') as pickle_file:
            stats = pickle.load(pickle_file)

        if data_target == 'sample':
            for i, sample in data.iterrows():
                for side in ['myo_left_readings', 'myo_right_readings']:
                    np_sample = np.array(sample[side])
                    data.at[i, side] = (np_sample - np_sample.mean(axis=0))/np_sample.std(axis=0)   
        elif data_target == 'channel':
            for side in ['myo_left_readings', 'myo_right_readings']:
                np_side_data = np.empty((0,8))
                lengths = []
                for sample in data[side]:
                    np_sample = np.array(sample)
                    lengths.append(np_sample.shape[0])
                    np_side_data = np.vstack((np_side_data, np_sample))
                
            for j in range(8):
                np_side_data[:,j] = (np_side_data[:,j] - np_side_data[:,j].mean()) / np_side_data[:,j].std()
                
            start = 0
            for i, l in enumerate(lengths):
                data.iat[i, data.columns.get_loc(side)] = np_side_data[start:start+l, :].tolist()
                start+=l                 

        return data

    def __scale__(self, data, data_target:str='channel_global') -> pd.DataFrame:
        stats = dict
        with open('stats.pkl', 'rb') as pickle_file:
            stats = pickle.load(pickle_file)

        if data_target == "sample":
            for i, sample in data.iterrows():
                for side in ['myo_left_readings', 'myo_right_readings']:
                    np_sample = np.array(sample[side])
                    # new_sample = (np_sample - np.amin(np_sample)) / (np.amax(np_sample) - np.amin(np_sample)) * 2 - 1
                    
                    sample_range = np.amax(np_sample) - np.amin(np_sample)
                    np_sample = np_sample / (sample_range / 2)
                    np_sample = np_sample - np.amin(np_sample) - 1
                    
                    data.at[i, side] = np_sample
        elif data_target == 'channel':
            for side in ['myo_left_readings', 'myo_right_readings']:
                np_side_data = np.empty((0,8))
                lengths = []
                for sample in data[side]:
                    np_sample = np.array(sample)
                    lengths.append(np_sample.shape[0])
                    np_side_data = np.vstack((np_side_data, np_sample))
                
            for j in range(8):
                x = np_side_data[:,j] / ((np.amax(np_side_data[:,j]) - np.amin(np_side_data[:,j])) / 2) 
                new_sample = x - np.amin(x) - 1
                np_side_data[:,j] = new_sample
            start = 0
            for i, l in enumerate(lengths):
                data.iat[i, data.columns.get_loc(side)] = np_side_data[start:start+l, :].tolist()
                start+=l    

        return data

    def __save_spectogram__(self, specgram_l, specgram_r, name):
        both_specs = [*specgram_l, *specgram_r]

        # trans = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.double())])
        
        #BW: 3s->8 5s -> 13 10s -> 26
        #BH: 3s->17 5s-> 17 10s -> 17
        
        # biggest_width=49
        
        all_spectrograms = []
        
        for spectrogram in both_specs:
            spectrogram = spectrogram.double()
    
            # if spectrogram.shape[1] < biggest_width:
            #     pad_left = (biggest_width - spectrogram.shape[1])//2
            #     pad_right = biggest_width - pad_left
            #     spectrogram = torch.nn.functional.pad(spectrogram, (pad_left,pad_right,0,0))  
                
            all_spectrograms.append(spectrogram)
            
        final_t = torch.stack(all_spectrograms, dim=0)
        
        torch.save(final_t, os.path.join("..","spectrograms", f"{name}.pt"))

    def __calculate_stats__(self) -> dict:
        split_train = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['split'], self.current_split_folder, self.split_files['train'])))
        samples = {}
        dataset_train = []

        for filename in os.listdir(os.path.join(self.FOLDERS['data'], self.current_emg_folder)):
            agent = filename[:5]
            samples_agent = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['data'], self.current_emg_folder, filename)))
            samples[agent] = samples_agent

        for i, row in split_train.iterrows():
            index = row['index']
            filename = row['file']
            agent = filename[:5]
            sample = samples[agent].loc[index, :]
            sample = sample.copy()
            dataset_train.append(sample)

        dataset_train = pd.DataFrame(dataset_train)
        left_readings = dataset_train['myo_left_readings']
        right_readings = dataset_train['myo_right_readings']

        left_readings = np.vstack([left_readings.values[i] for i in range(len(left_readings))])
        right_readings = np.vstack([right_readings.values[i] for i in range(len(right_readings))])

        stats = {
            'left_max_value': left_readings.max(axis=0).reshape(1, 8), 
            'left_min_value': left_readings.min(axis=0).reshape(1, 8),
            'right_max_value': right_readings.max(axis=0).reshape(1, 8), 
            'right_min_value': right_readings.min(axis=0).reshape(1, 8),
            'left_mean': left_readings.mean(axis=0).reshape(1, 8),
            'right_mean': right_readings.mean(axis=0).reshape(1, 8),
            "left_std": left_readings.std(axis=0).reshape(1, 8),
            "right_std": right_readings.std(axis=0).reshape(1, 8),
            "g_min": min(left_readings.min(), right_readings.min()),
            "g_max": max(left_readings.max(), right_readings.max()),
            "g_mean": (left_readings.mean()+right_readings.mean())/2.,
            "g_std": np.vstack((left_readings, right_readings)).reshape(-1,).std()
        }

        with open('stats.pkl', 'wb') as pickle_file:
            pickle.dump(stats, pickle_file)

    def delete_temps(self) -> None:
        folders = []
        
        for filename in os.listdir(os.path.join(os.getcwd(), '..', 'spectrograms')):
            file_path = os.path.join(os.getcwd(), '..', 'spectrograms', filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
        
        for item in os.listdir(self.FOLDERS['data']):
            if os.path.isdir(os.path.join(self.FOLDERS['data'], item)):
                folders.append(item)

        for folder in folders:   
            if folder == 'base':
                continue
            for filename in os.listdir(os.path.join(self.FOLDERS['data'], folder)):
                os.remove(os.path.join(self.FOLDERS['data'], folder, filename))
            os.rmdir(os.path.join(self.FOLDERS['data'], folder))

        folders = []
        for item in os.listdir(self.FOLDERS['split']):
            if os.path.isdir(os.path.join(self.FOLDERS['split'], item)):
                folders.append(item)

        for folder in folders:   
            if folder == 'base' or folder == 'rgb-video':
                continue
            for filename in os.listdir(os.path.join(self.FOLDERS['split'], folder)):
                os.remove(os.path.join(self.FOLDERS['split'], folder, filename))
            os.rmdir(os.path.join(self.FOLDERS['split'], folder))

    def augment_dataset(self, time_interval:int=10, num_chunks:int=20, sampling_rate:float=10., type_padding:str='zeros') -> None:
        next_folder = f'step{self.current_step}-augment'
        if not os.path.exists(os.path.join(self.FOLDERS['data'], next_folder)):
            os.makedirs(os.path.join(self.FOLDERS['data'], next_folder))

        for filename in os.listdir(os.path.join(self.FOLDERS['data'], self.current_emg_folder)):
            data = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['data'], self.current_emg_folder, filename)))
            data = data[data['description'] != 'calibration']
            new_data = []

            for i, row in data.iterrows():
                chunks = self.__multiply_actions__(
                    left_timestamps=row['myo_left_timestamps'], 
                    left_readings=row['myo_left_readings'], 
                    right_timestamps=row['myo_right_timestamps'], 
                    right_readings=row['myo_right_readings'], 
                    time_interval=time_interval, 
                    num_chunks=num_chunks, 
                    sampling_rate=sampling_rate,
                    type_padding='zeros'
                )
                left_timestamps_chunks, right_timestamps_chunks, left_readings_chunks, right_readings_chunks = chunks

                for c in range(len(left_timestamps_chunks)):
                    # handle old descriptions
                    if data.loc[i, 'description'] in self.DESCRIPTIONS_CONVERSION_DICT:
                        data.at[i, 'description'] = self.DESCRIPTIONS_CONVERSION_DICT[data.loc[i, 'description']]

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

            # Convert the list of dictionaries to a DataFrame
            new_data = pd.DataFrame(new_data)
            new_data.to_pickle(os.path.join(self.FOLDERS['data'], next_folder, filename))

        self.current_emg_folder = next_folder
        # Update the split files with the new augmented dataset
        self.__update_split_files__()
        self.current_step += 1

        print(f'Dataset was correctly augmented')

    def resample(self, sampling_rate:float=10., data_target = "") -> None:
        try:
            #sampling_interval = 1/sampling_rate

            next_folder = f'step{self.current_step}-resample'
            if not os.path.exists(os.path.join(self.FOLDERS['data'], next_folder)):
                os.makedirs(os.path.join(self.FOLDERS['data'], next_folder))
            self.current_step +=1
            
            for filename in os.listdir(os.path.join(self.FOLDERS['data'], self.current_emg_folder)):
                dataframe = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['data'], self.current_emg_folder, filename)))
                dataframe = dataframe[dataframe['description'] != 'calibration']
    
                for i, row in dataframe.iterrows():
                    timestamps_sx = row['myo_left_timestamps']
                    timestamps_dx = row['myo_right_timestamps']
                    readings_sx = np.array(row['myo_left_readings'])
                    readings_dx = np.array(row['myo_right_readings'])

                    num_samples_new_sx = int(round(sampling_rate * (timestamps_sx[-1] - timestamps_sx[0])))
                    num_samples_new_dx = int(round(sampling_rate * (timestamps_dx[-1] - timestamps_dx[0])))
                    
                    new_timestamps_sx = np.linspace(timestamps_sx[0], timestamps_sx[-1], num=num_samples_new_sx + 1, endpoint=True)
                    new_timestamps_dx = np.linspace(timestamps_dx[0], timestamps_dx[-1], num=num_samples_new_dx + 1, endpoint=True)
                    
                    fn_interpolate_sx = interpolate.interp1d(timestamps_sx, readings_sx, axis=0, kind='linear', fill_value='extrapolate')
                    fn_interpolate_dx = interpolate.interp1d(timestamps_dx, readings_dx, axis=0, kind='linear', fill_value='extrapolate')

                    new_readings_sx = fn_interpolate_sx(new_timestamps_sx)
                    new_readings_dx = fn_interpolate_dx(new_timestamps_dx)

                    if np.any(np.isnan(new_readings_sx)):
                        timesteps_have_nan = np.any(np.isnan(new_readings_sx), axis=tuple(np.arange(1,np.ndim(new_readings_sx))))
                        new_readings_sx[np.isnan(new_readings_sx)] = 0
                    if np.any(np.isnan(new_readings_dx)):
                        timesteps_have_nan = np.any(np.isnan(new_readings_dx), axis=tuple(np.arange(1,np.ndim(new_readings_dx))))
                        new_readings_dx[np.isnan(new_readings_dx)] = 0

                    dataframe.at[i, 'myo_left_timestamps'] = new_timestamps_sx
                    dataframe.at[i, 'myo_right_timestamps'] = new_timestamps_dx
                    dataframe.at[i, 'myo_left_readings'] = new_readings_sx
                    dataframe.at[i, 'myo_right_readings'] = new_readings_dx

                dataframe.to_pickle(os.path.join(self.FOLDERS['data'], next_folder, filename))

        except Exception as e:
            print(f"Error: {e}")

        self.current_emg_folder = next_folder
        print(f'Dataset was correctly resampled')

    def pre_processing(self, data_target:str, operations:list, cut_frequency:float, filter_order:int) -> None:
        map_functions = {
            'filter': lambda data: self.__low_pass_filter__(data, cut_frequency, filter_order, data_target),
            'scale': lambda data: self.__scale__(data, data_target),
            'normalize': lambda data: self.__normalize__(data, data_target),
        }

        next_folder = f'step{self.current_step}-preproc'
        if not os.path.exists(os.path.join(self.FOLDERS['data'], next_folder)):
            os.makedirs(os.path.join(self.FOLDERS['data'], next_folder))
        self.current_step += 1

        for filename in os.listdir(os.path.join(self.FOLDERS['data'], self.current_emg_folder)):
            data = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['data'], self.current_emg_folder, filename)))
            data = data[data['description'] != 'calibration']

            data['myo_left_readings'] = np.abs(data['myo_left_readings'].values)
            data['myo_right_readings'] = np.abs(data['myo_right_readings'].values)

            data.to_pickle(os.path.join(self.FOLDERS['data'], next_folder, filename))            

        self.current_emg_folder = next_folder
        self.__calculate_stats__()
        for filename in os.listdir(os.path.join(self.FOLDERS['data'], self.current_emg_folder)):
            data = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['data'], self.current_emg_folder, filename)))
            for op in operations:
                data = map_functions[op](data)
                
            data.to_pickle(os.path.join(self.FOLDERS['data'], self.current_emg_folder, filename))

        print("Dataset was correctly preprocessed")            

    def generate_spectograms(self, save_spectrograms:bool=True) -> None:
        # backup_spectrograms = {'Get/replace items from refrigerator/cabinets/drawers': 'S00_2_0', 'Peel a cucumber': 'S00_2_15', 'Slice a cucumber': 'S00_2_43', 'Peel a potato': 'S00_2_72', 'Slice a potato': 'S00_2_104', 'Slice bread': 'S00_2_134', 'Spread almond butter on a bread slice': 'S00_2_165', 'Spread jelly on a bread slice': 'S00_2_180', 'Open/close a jar of almond butter': 'S00_2_189', 'Pour water from a pitcher into a glass': 'S00_2_201', 'Clean a plate with a sponge': 'S00_2_224', 'Clean a plate with a towel': 'S00_2_236', 'Clean a pan with a sponge': 'S00_2_243', 'Clean a pan with a towel': 'S00_2_251', 'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 'S00_2_260', 'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 'S00_2_282', 'Stack on table: 3 each large/small plates, bowls': 'S00_2_304', 'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 'S00_2_315', 'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils': 'S00_2_350', 'Clear cutting board': 'S02_2_48'}

        backup_spectrograms = {}
        
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

        self.specto_folder = f'step{self.current_step}-specto'
        if not os.path.exists(os.path.join(self.FOLDERS['data'], self.specto_folder)):
            os.makedirs(os.path.join(self.FOLDERS['data'], self.specto_folder))
        self.current_step += 1

        file_list = os.listdir(os.path.join(self.FOLDERS['data'], self.current_emg_folder))
        print('Saving spectrograms...')
        print(f"\r0/{len(file_list)}", end='')
        for i_file, filename in enumerate(file_list):
            cur_values = []
            emg_annotations = pd.read_pickle(os.path.join(self.FOLDERS['data'], self.current_emg_folder, filename))
            for sample_no in range(len(emg_annotations)):
                signal_l = torch.Tensor(np.array(emg_annotations.iloc[sample_no].myo_left_readings))
                signal_r = torch.Tensor(np.array(emg_annotations.iloc[sample_no].myo_right_readings))
                label = emg_annotations.iloc[sample_no].description
                file_name_prefix = os.path.splitext(filename)[0]
                name = f"{file_name_prefix}_{sample_no}"
                try:
                    freq_signal_l = compute_spectrogram(signal_l)
                    freq_signal_r = compute_spectrogram(signal_r)
                    if save_spectrograms:
                        self.__save_spectogram__(freq_signal_l, freq_signal_r, name)
                    new_row_data = [name, label]
                    if label not in backup_spectrograms.keys():
                       backup_spectrograms[label] = name
                    cur_values.append(new_row_data)
                except RuntimeError:
                    new_row_data = [backup_spectrograms[label], label]
                    cur_values.append(new_row_data)

            cur_df = pd.DataFrame(cur_values, columns=['file','description'])
            cur_df.to_pickle(os.path.join(self.FOLDERS['data'], self.specto_folder, filename))
            print(f"\r{i_file+1}/{len(file_list)}", end='')

        print(f'\rDataset specto built')

    def generate_rgb(self) -> None:
        self.rgb_folder = f'step{self.current_step}-rgb'
        if not os.path.exists(os.path.join(self.FOLDERS['data'], self.rgb_folder)):
            os.makedirs(os.path.join(self.FOLDERS['data'], self.rgb_folder))
        self.current_step += 1
        for filename in os.listdir(os.path.join(self.FOLDERS['data'], self.current_emg_folder)):
            try: 
                data = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['data'], self.current_emg_folder, filename)))
                new_data = []

                for i, row in data.iterrows():
                    new_row = {
                        'start_timestamp': row['start'],
                        'stop_timestamp': row['stop'],
                        'start_frame': row['start'] * self.FRAME_RATE,
                        'stop_frame': row['stop'] * self.FRAME_RATE,
                        'verb': row['description'],
                        'verb_class': self.DESCRIPTIONS_TO_LABELS.index(row['description']),
                    }
                    new_data.append(new_row)

                # Convert the list of dictionaries to a DataFrame
                new_data = pd.DataFrame(new_data)

                # Save the DataFrame to a pickle file
                new_data.to_pickle(os.path.join(self.FOLDERS['data'], self.rgb_folder, filename))
            except Exception as e:
                print(f"Error: {e}")

        print(f'Dataset RGB built')

    def merge_pickles(self) -> None:
        agents = {}
        for agent in self.__get_agents__():
            agents[agent] = {
                'rgb_file': None,
                'specto_file': None,
                'emg_file': None,
            }

        for filename in os.listdir(os.path.join(self.FOLDERS['data'], self.rgb_folder)):
            agent = filename[:5]
            agents[agent]['rgb_file'] = filename

        for filename in os.listdir(os.path.join(self.FOLDERS['data'], self.specto_folder)):
            agent = filename[:5]
            agents[agent]['specto_file'] = filename

        for filename in os.listdir(os.path.join(self.FOLDERS['data'], self.current_emg_folder)):
            agent = filename[:5]
            agents[agent]['emg_file'] = filename

        next_folder = f'step{self.current_step}-merge'
        if not os.path.exists(os.path.join(self.FOLDERS['data'], next_folder)):
            os.makedirs(os.path.join(self.FOLDERS['data'], next_folder))
        self.current_step += 1

        for agent, files in agents.items():
            emg = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['data'], self.current_emg_folder, files['emg_file'])))
            rgb = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['data'], self.rgb_folder, files['rgb_file'])))
            specto = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['data'], self.specto_folder, files['specto_file'])))

            # remove t0 from timestamps and frame nubers T0
            # >>> min(emg['start'])
            # 1655239114.183343 ====> calibration start
            
            t0 = min(pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['data'], 'base', f'{agent}.pkl')))['start'])

            emg['start'] = emg['start'].map(lambda x: self.__remove_t0_time__(x, t0=t0))
            emg['stop'] = emg['stop'].map(lambda x: self.__remove_t0_time__(x, t0=t0))
            emg['myo_left_timestamps'] = emg['myo_left_timestamps'].map(lambda x: self.__remove_t0_time__(x, t0=t0))
            emg['myo_right_timestamps'] = emg['myo_right_timestamps'].map(lambda x: self.__remove_t0_time__(x, t0=t0))
            rgb['start_frame'] = rgb['start_frame'].map(lambda x: self.__remove_t0_frame__(x, t0=t0))
            rgb['stop_frame'] = rgb['stop_frame'].map(lambda x: self.__remove_t0_frame__(x, t0=t0))

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

            final.to_pickle(os.path.join(self.FOLDERS['data'], next_folder, f'{agent}.pkl'))

        self.current_emg_folder = next_folder
        print("Pickles were correctly merged")

    def balance_splits(self, train_split_proportion:float=.05) -> None:
        split_train = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['split'], self.current_split_folder, self.split_files['train'])))
        split_test = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['split'], self.current_split_folder, self.split_files['test'])))
        samples = {}
        dataset_train = []
        dataset_test = []
        n_samples_x_class = {'train': np.zeros(20), 'test': np.zeros(20)}
        num_classes = 20

        for filename in os.listdir(os.path.join(self.FOLDERS['data'], self.current_emg_folder)):
            agent = filename[:5]
            samples_agent = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['data'], self.current_emg_folder, filename)))
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

        for c in range(num_classes):
            candidates_to_move = dataset_train[dataset_train['label'] == c]       
            if n_samples_x_class['test'][c] / n_samples_x_class['train'][c] <= train_split_proportion:
                print(f'balancing class {c}...')
                n_to_move = math.ceil(train_split_proportion*n_samples_x_class['train'][c] - n_samples_x_class['test'][c])
                to_move = candidates_to_move.head(n_to_move)

                print(f'moving {n_to_move} samples from tarin to test:')
                print(to_move)

                n_samples_x_class['train'][c] -= n_to_move
                n_samples_x_class['test'][c] += n_to_move

                for i, row in to_move.iterrows():
                    row_to_move = split_train[(split_train['file']==row['from_file']) & (split_train['index']==i)]
                    split_train = split_train.drop(row_to_move.index)
                    split_test = pd.concat([split_test, row_to_move], ignore_index=True)

        self.current_split_folder = f'step{self.current_step}-balance'
        if not os.path.exists(os.path.join(self.FOLDERS['split'], self.current_split_folder)):
            os.makedirs(os.path.join(self.FOLDERS['split'], self.current_split_folder))
        self.current_step += 1
        split_train.to_pickle(os.path.join(self.FOLDERS['split'], self.current_split_folder, self.split_files['train']))
        split_test.to_pickle(os.path.join(self.FOLDERS['split'], self.current_split_folder, self.split_files['test']))
        
        print('\n\nNumber of samples per class TRAIN/TEST\n')
        train_cardinality = split_train.groupby('description').size() 
        print("TRAIN")
        print(train_cardinality, end ="\n\n")
        test_cardinality = split_test.groupby('description').size() 
        print("TEST")
        print(test_cardinality)

        print('Split files were correctly balanced')

if __name__ == '__main__':
    resampling_rate=10.

    processing = ProcessEmgDataset()
    processing.delete_temps()
    processing.pre_processing(data_target="sample", operations=['filter', 'scale'], cut_frequency=5., filter_order=5)
    processing.resample(sampling_rate=10.)
    processing.augment_dataset(time_interval=5.)
    processing.generate_spectograms(save_spectrograms=True)
    processing.generate_rgb()
    processing.merge_pickles()
    processing.balance_splits(train_split_proportion=.05)
