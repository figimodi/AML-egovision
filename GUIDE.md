# ActionSense

## Prepare data
1. Launch `python process_emg.py`
2. from command line:
```
mkdir action-net/rgb-video
python
>>> import pandas as pd
>>> df = pd.DataFrame(pd.read_pickle('action-net/step7-balance/ActionNet_train.pkl'))
>>> df = df[df['file']=='S04_1.pkl']
>>> df.to_pickle('action-net/step7-balance/ActionNet_train.pkl')
>>> df = pd.DataFrame(pd.read_pickle('action-net/step7-balance/ActionNet_test.pkl'))
>>> df = df[df['file']=='S04_1.pkl']
>>> df.to_pickle('action-net/step7-balance/ActionNet_test.pkl')
```
3. Launch the following code inside action-net/rgb-video:
```py
def balance_splits(train_split_proportion:float=.05) -> None:
    split_train = pd.DataFrame(pd.read_pickle('ActionNet_train.pkl'))
    split_test = pd.DataFrame(pd.read_pickle('ActionNet_test.pkl'))
    samples = {}
    dataset_train = []
    dataset_test = []
    n_samples_x_class = {'train': np.zeros(20), 'test': np.zeros(20)}
    num_classes = 20

    samples_agent = pd.DataFrame(pd.read_pickle('./../../emg/step6-merge/S04_1.pkl'))
    samples = samples_agent

    for i, row in split_train.iterrows():
        index = row['index']
        filename = row['file']
        sample = samples.loc[index, :]
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
        sample = samples.loc[index, :]
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

    split_train.to_pickle('ActionNet_train.pkl')
    split_test.to_pickle('ActionNet_test.pkl')
    
    print('\n\nNumber of samples per class TRAIN/TEST\n')
    train_cardinality = split_train.groupby('description').size() 
    print("TRAIN")
    print(train_cardinality, end ="\n\n")
    test_cardinality = split_test.groupby('description').size() 
    print("TEST")
    print(test_cardinality)

    print('Split files were correctly balanced')
```

## Train EMG
1. Into configs/train_emg.yaml, set split_path to action-net/step7-balance
2. Launch `python train_classifier_emg.py split=train action=train name=EMG_pretrained modality=EMG`

## Train RGB
1. Into configs/train_emg.yaml, set split_path to action-net/rgb-video
2. Launch `python train_classifier_emg.py split=train action=train name=RGB_pretrained modality=RGB`

## Train specto
1. Into configs/train_emg.yaml, set split_path to action-net/step7-balance
2. Launch `python train_classifier_emg.py split=train action=train name=specto_pretrained modality=specto`

## Train fusions
1. Into train_classifier_fusion.py set fusion_modalities to the array of modalities you want to use (EMG, RGB, specto)
2. Into configs/train_fusion.yaml set modalities.fusion.models.fusion.name to the correspective model for the fusion chosing from: FusionClassifierEMGRGB, FusionClassifierRGBspecto, FusionClassifierEMGspecto and FusionClassifierEMGRGBspecto
3. Into tasks/action_recognition_task_fusion,py modify line 74 to pass the data corresponding to the modalities of the fusion, examples are:
`logits, feat = self.task_models['fusion'](emg=data['EMG'], rgb=data['RGB'], specto=data['specto'], **kwargs)` for a combination of emg, rgb and specto
`logits, feat = self.task_models['fusion'](emg=data['EMG'], specto=data['specto'], **kwargs)` for a combination of emg and specto
4. Launch `python.exe .\train_classifier_fusion.py action=train name=fusion_EMG_RGB split=train modality=fusion`

## Launch late-fusion
1. Launch `python late_fusion.py split=train action=train name=late modality=EMG`