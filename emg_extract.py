import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def emg_adjust_features(file_path: str, *, cut_frequency: float = 5.0, filter_order: int = 4):
    data = pd.DataFrame(pd.read_pickle(file_path))
    to_read = ['myo_right', 'myo_left']

    for side in to_read:
        for i, _ in data.iterrows():
            if i != 0:
                # Rectify channels
                row = abs(data.loc[i, side + '_readings'])
                times = data.loc[i, side + '_timestamps']
                
                # Low-pass filter 5 Hz
                fs = times.size / (times[-1]-times[0])

                nyq = 0.5 * fs
                wn = cut_frequency / nyq
                
                b, a = butter(filter_order, wn, 'low', analog=False)
                filtered = filtfilt(b, a, row.T, padlen=1).T
                
                # Normalize in [-1, 1]
                scaler = MinMaxScaler(feature_range=(-1, 1))
                filtered = scaler.fit_transform(filtered)
                
                # Forearm activation
                activation = filtered.sum(1)
                
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