import pandas as pd
import os
import math
import numpy as np


if __name__ == '__main__':
    train_split_proportion=0.2
    split_train = pd.DataFrame(pd.read_pickle(os.path.join('action-net/rgb-video', 'ActionNet_train.pkl')))
    split_test = pd.DataFrame(pd.read_pickle(os.path.join('action-net/rgb-video', 'ActionNet_test.pkl')))
    samples = {}
    dataset_train = []
    dataset_test = []
    n_samples_x_class = {'train': np.zeros(20), 'test': np.zeros(20)}
    num_classes = 20

    for filename in os.listdir('emg/step6-merge'):
        agent = filename[:5]
        samples_agent = pd.DataFrame(pd.read_pickle(os.path.join('emg/step6-merge', filename)))
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

    split_train.to_pickle(os.path.join('action-net/rgb-video', 'ActionNet_train.pkl'))
    split_test.to_pickle(os.path.join('action-net/rgb-video', 'ActionNet_test.pkl'))
    
    print('\n\nNumber of samples per class TRAIN/TEST\n')
    train_cardinality = split_train.groupby('description').size() 
    print("TRAIN")
    print(train_cardinality, end ="\n\n")
    test_cardinality = split_test.groupby('description').size() 
    print("TEST")
    print(test_cardinality)

    print('Split files were correctly balanced')