# ActionSense
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