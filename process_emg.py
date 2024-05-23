from scipy import signal, interpolate
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchaudio.functional as F
import torchaudio.transforms as T
import torch
import os
import math
import imageio
import cv2
import librosa
import pickle

class ProcessEmgDataset():
    def __init__(self, emg_checkpoint_folder:str=None, split_checkpoint_folder:str=None, rgb_folder:str=None, specto_folder:str=None):
        self.FOLDERS = {
            'data': 'emg',
            'split': 'action-net',
        }

        self.DESCRIPTIONS_TO_LABELS = [
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

    def __trim_samples__(self, left_timestamps:list, left_readings:list, right_timestamps:list, right_readings:list, time_interval:int=5) -> Tuple[list]:
         # Initialize empty lists to store the chunks and readings
        left_chunks = [[]]
        right_chunks = [[]]
        left_reading_chunks = [[]]
        right_reading_chunks = [[]]

        if left_timestamps.shape == () or right_timestamps.shape == ():
            return [None for _ in range(4)]

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

    def __remove_t0_time__(self, value: float, t0: float) -> float:
        if type(value)==list:
            return [t - t0 for t in value]
        else:
            return value - t0

    def __remove_t0_frame__(self, value: float, t0: float) -> int:
        # remove t0. add 1 because the first frame is 0000001 and not 0000000
        return int(value - (t0*self.FRAME_RATE) + 1)

    def __pad_item__(self, sample, type_padding:str='mean', size:int=0) -> None:
        left_readings = sample['myo_left_readings']
        right_readings = sample['myo_right_readings']

        readings = {'myo_left_readings': left_readings, 'myo_right_readings': right_readings}

        for key, value in readings.items():
            original_length = len(value)
            diff = size - original_length

            left_padding_lenght = diff // 2
            right_padding_lenght = diff - left_padding_lenght

            average_value = sum(value)/len(value)

            pad_value = np.array(average_value).reshape(1, 8)

            # Pad the list with zeros on both sides
            if left_padding_lenght > 0:
                if type_padding == 'noise':
                    left_padding = pad_value + np.random.normal(0, 1, (left_padding_lenght, 8))
                elif type_padding == 'zeros':
                    left_padding = np.zeros(8).repeat(right_padding_lenght, axis=0)
                else:
                    left_padding = pad_value.repeat(left_padding_lenght, axis=0)
                value = np.concatenate((left_padding, value), axis=0)
            if right_padding_lenght > 0:
                if type_padding == 'noise':
                    right_padding = pad_value + np.random.normal(0, 1, (right_padding_lenght, 8))
                elif type_padding == 'zeros':
                    right_padding = np.zeros(8).repeat(right_padding_lenght, axis=0)
                else:
                    right_padding = pad_value.repeat(right_padding_lenght, axis=0)
                value = np.concatenate((value, right_padding), axis=0)
            sample[key] = value

        return sample

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

    def __low_pass_filter__(self, data, fs:float=160., cut_frequency:float=5., filter_order:int=2) -> pd.DataFrame:
        nyquist_freq = fs * 0.5  # Nyquist frequency
        normalized_cutoff = cut_frequency / nyquist_freq
        b, a = signal.butter(filter_order, normalized_cutoff, btype='lowpass')  # Butterworth filter

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
        if data_target == 'channel':
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
        elif data_target == 'channel_global':
            stats = dict
            with open('stats.pkl', 'rb') as pickle_file:
                stats = pickle.load(pickle_file)
            
            sx_data = data['myo_left_readings'].values
            for i in range(len(sx_data)):
                sx_data[i] = (sx_data[i] - stats['left_mean'])  / stats['left_std']

            dx_data = data['myo_right_readings'].values
            for i in range(len(dx_data)):
                dx_data[i] = (dx_data[i] - stats['right_mean'])  / stats['right_std']

            data['myo_left_readings'] = sx_data
            data['myo_right_readings'] = dx_data

            return data
        elif data_target == "global":
            #copy from data (dataframe) to np_side_data(np array)
            np_side_data = {}
            for side in ['myo_left_readings', 'myo_right_readings']:
                np_side_data_app = np.empty((0,8))
                lengths = []

                for sample in data[side]:
                    np_sample = np.array(sample)
                    lengths.append(np_sample.shape[0])
                    np_side_data[side] = np.vstack((np_side_data_app, np_sample))
            
            mean = stats["g_mean"]
            std = stats["g_std"]
        
            np_side_data[side] = (np_side_data[side] - mean) / std  
            
            start = 0
            for i, l in enumerate(lengths):
                data.iat[i, data.columns.get_loc(side)] = np_side_data[side][start:start+l, :].tolist()
                start+=l               
        
            return data

    def __scale__(self, data, data_target:str='channel_global') -> pd.DataFrame:
        if data_target == 'channel':
            for side in ['myo_left_readings', 'myo_right_readings']:
                np_side_data = np.empty((0,8))
                lengths = []
                for sample in data[side]:
                    np_sample = np.array(sample)
                    lengths.append(np_sample.shape[0])
                    np_side_data = np.vstack((np_side_data, np_sample))
                
            for j in range(8):
                np_side_data[:,j] = (np_side_data[:,j] - np_side_data[:,j].min()) / (np_side_data[:,j].max() - np_side_data[:,j].min()) * 2 - 1 
            start = 0
            for i, l in enumerate(lengths):
                data.iat[i, data.columns.get_loc(side)] = np_side_data[start:start+l, :].tolist()
                start+=l    

            return data
        elif data_target == 'channel_global':
            stats = dict
            with open('stats.pkl', 'rb') as pickle_file:
                stats = pickle.load(pickle_file)
            
            sx_data = data['myo_left_readings'].values
            for i in range(len(sx_data)):
                sx_data[i] = (sx_data[i] - stats['left_min_value'])  / (stats['left_max_value'] - stats['left_min_value']) * 2 - 1

            dx_data = data['myo_right_readings'].values
            for i in range(len(dx_data)):
                dx_data[i] = (dx_data[i] - stats['right_min_value'])  / (stats['right_max_value'] - stats['right_min_value']) * 2 - 1

            data['myo_left_readings'] = sx_data
            data['myo_right_readings'] = dx_data

            return data
        elif data_target == "global":
            stats = dict
            with open('stats.pkl', 'rb') as pickle_file:
                stats = pickle.load(pickle_file)
            
            #copy from data (dataframe) to np_side_data(np array)
            np_side_data = {}
            for side in ['myo_left_readings', 'myo_right_readings']:
                np_side_data_app = np.empty((0,8))
                lengths = []
                
                for sample in data[side]:
                    np_sample = np.array(sample)
                    lengths.append(np_sample.shape[0])
                    np_side_data[side] = np.vstack((np_side_data_app, np_sample))

            abs_min = stats["g_min"]
            abs_max = stats["g_max"]
        
            np_side_data[side] = (np_side_data[side] - abs_min) / (abs_max - abs_min) * 2 - 1    
                
            start = 0
            for i, l in enumerate(lengths):
                data.iat[i, data.columns.get_loc(side)] = np_side_data[side][start:start+l, :].tolist()
                start+=l               
        
            return data

    def __save_spectogram__(self, specgram_l, specgram_r, name, resize_factor=.25) -> None:
        both_specs = [*specgram_l, *specgram_r]
    
        for i in range(len(both_specs)):
            plt.figure()
            
            plt.imshow(librosa.power_to_db(both_specs[i]), origin="lower", aspect="auto")
            plt.axis('off')
            plt.gca().set_position([0, 0, 1, 1])
            
            # plt.savefig(f"../spectrograms/{name}_{i}")
            
            fig = plt.gcf()
            
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # Resize
            image_from_plot = cv2.resize(image_from_plot, dsize=None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

            # Save as an image (you can choose the format based on your needs)

            if not os.path.exists('../spectrograms'):
                os.makedirs('../spectrograms')
            imageio.imwrite(f"../spectrograms/{name}_{i}.png", image_from_plot)
            plt.close()

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
            'left_max_value': left_readings.max(axis=0).reshape(8, 1), 
            'left_min_value': left_readings.min(axis=0).reshape(8, 1),
            'right_max_value': right_readings.max(axis=0).reshape(8, 1), 
            'right_min_value': right_readings.min(axis=0).reshape(8, 1),
            'left_mean': left_readings.mean(axis=0).reshape(8, 1),
            'right_mean': right_readings.mean(axis=0).reshape(8, 1),
            "left_std": left_readings.std(axis=0).reshape(8, 1),
            "right_std": right_readings.std(axis=0).reshape(8, 1),
            "g_min": min(left_readings.min(), right_readings.min()),
            "g_max": max(left_readings.max(), right_readings.max()),
            "g_mean": (left_readings.mean()+right_readings.mean())/.2,
            "g_std": np.vstack(left_readings, right_readings).reshape(-1,).std()
        }

        with open('stats.pkl', 'wb') as pickle_file:
            pickle.dump(stats, pickle_file)

    def delete_temps(self) -> None:
        folders = []
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
            if folder == 'base':
                continue
            for filename in os.listdir(os.path.join(self.FOLDERS['split'], folder)):
                os.remove(os.path.join(self.FOLDERS['split'], folder, filename))
            os.rmdir(os.path.join(self.FOLDERS['split'], folder))

    def augment_dataset(self, time_interval:int=5) -> None:
        next_folder = f'step{self.current_step}-augment'
        if not os.path.exists(os.path.join(self.FOLDERS['data'], next_folder)):
            os.makedirs(os.path.join(self.FOLDERS['data'], next_folder))

        for filename in os.listdir(os.path.join(self.FOLDERS['data'], self.current_emg_folder)):
            data = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['data'], self.current_emg_folder, filename)))
            data = data[data['description'] != 'calibration']
            new_data = []

            for i, row in data.iterrows():
                chunks = self.__trim_samples__(row['myo_left_timestamps'], row['myo_left_readings'], row['myo_right_timestamps'], row['myo_right_readings'], time_interval)
                if any(item is None for item in chunks):
                    continue

                left_timestamps_chunks, right_timestamps_chunks, left_readings_chunks, right_readings_chunks = chunks
                if len(left_timestamps_chunks) < len(right_timestamps_chunks):
                    right_timestamps_chunks = right_timestamps_chunks[:-1]
                    right_readings_chunks = right_readings_chunks[:-1]
                if len(left_timestamps_chunks) > len(right_timestamps_chunks):
                    left_timestamps_chunks = left_timestamps_chunks[:-1]
                    left_readings_chunks = left_readings_chunks[:-1]

                n_chunks = len(left_timestamps_chunks)

                for c in range(n_chunks):
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

    def resample(self, sampling_rate:float=10.) -> None:
        try:
            sampling_interval = 1/sampling_rate

            next_folder = f'step{self.current_step}-resample'
            if not os.path.exists(os.path.join(self.FOLDERS['data'], next_folder)):
                os.makedirs(os.path.join(self.FOLDERS['data'], next_folder))
            self.current_step +=1
            
            for filename in os.listdir(os.path.join(self.FOLDERS['data'], self.current_emg_folder)):
                dataframe = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['data'], self.current_emg_folder, filename)))
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

                dataframe.to_pickle(os.path.join(self.FOLDERS['data'], next_folder, filename))

        except Exception as e:
            print(f"Error: {e}")

        self.current_emg_folder = next_folder
        print(f'Dataset was correctly resampled')

    def padding(self, type_padding:str='mean') -> None:
        samples = {}
        max_sizes = {}

        next_folder = f'step{self.current_step}-padding'
        if not os.path.exists(os.path.join(self.FOLDERS['data'], next_folder)):
            os.makedirs(os.path.join(self.FOLDERS['data'], next_folder))
        self.current_step += 1

        for filename in os.listdir(os.path.join(self.FOLDERS['data'], self.current_emg_folder)):
            agent = filename[:5]
            samples_agent = pd.DataFrame(pd.read_pickle(os.path.join(self.FOLDERS['data'], self.current_emg_folder, filename)))
            samples[agent] = samples_agent
            max_sizes[agent] = max([max(len(sample['myo_left_readings']), len(sample['myo_right_readings'])) for _, sample in samples[agent].iterrows()])
        
        max_lenght = max([max_size for max_size in max_sizes.values()])

        for agent, dataframe in samples.items():
            dataframe = dataframe.apply(lambda row: self.__pad_item__(row, type_padding, max_lenght), axis=1)
            dataframe.to_pickle(os.path.join(self.FOLDERS['data'], next_folder, f"{agent}.pkl"))

        self.current_emg_folder = next_folder

        print(f"Dataset was correclty padded")

    def pre_processing(self, data_target:str='channel_global', operations:list=['filter', 'scale', 'normalize'], fs:float=160., cut_frequency:float=5., filter_order:int=2) -> None:
        map_functions = {
            'filter': lambda data: self.__low_pass_filter__(data, fs, cut_frequency, filter_order),
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

        self.specto_folder = f'step{self.current_step}-specto'
        if not os.path.exists(os.path.join(self.FOLDERS['data'], self.specto_folder)):
            os.makedirs(os.path.join(self.FOLDERS['data'], self.specto_folder))
        self.current_step += 1

        for filename in os.listdir(os.path.join(self.FOLDERS['data'], self.current_emg_folder)):
            cur_values = []
            agent = filename[:5]
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
                    cur_values.append(new_row_data)
                except RuntimeError:
                    new_row_data = [backup_spectrograms[label], label]
                    cur_values.append(new_row_data)

            cur_df = pd.DataFrame(cur_values, columns=['file','description'])
            cur_df.to_pickle(os.path.join(self.FOLDERS['data'], self.specto_folder, filename))

        print(f'Dataset specto built')

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
            t0 = min(emg['start'])

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

        print('Split files were correctly balanced')

if __name__ == '__main__':
    processing = ProcessEmgDataset()
    processing.delete_temps()
    processing.pre_processing(norm_method="channel", operations=['filter', 'normalize', 'scale'], fs=160., cut_frequency=5., filter_order=2)
    processing.resample(sampling_rate=10.)
    processing.augment_dataset(time_interval=5)
    processing.generate_spectograms(save_spectrograms=False)
    processing.padding(type_padding='mean')
    processing.generate_rgb()
    processing.merge_pickles()
    processing.balance_splits()
