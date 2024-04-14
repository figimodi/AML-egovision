import glob
from abc import ABC
import random
import pandas as pd
from .epic_record import EpicVideoRecord
from .action_record import ActionRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
from utils.logger import logger

import numpy as np

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


class ActionSenseDataset(data.Dataset, ABC):
    def __init__(self, mode, modalities, extract_features=False, sampling='dense', n_frames=16) -> None:
        file_name = f'./action-net/ActionNet_{mode}_augmented.pkl'
        self.split_file = pd.DataFrame(pd.read_pickle(file_name))
        self.modalities = modalities
        self.extract_features = extract_features
        self.model_features = None
        self.samples_dict = {}
        self.video_list = []

        # load all the samples
        for filename in os.listdir('emg/'):
            if os.path.isfile(os.path.join('/emg', filename)) and 'actionnet' in filename:
                samples = pd.DataFrame(pd.read_pickle(os.path.join('/emg', filename)))
                agent = filename[:5]
                self.samples_dict[agent] = samples

        if self.extract_features:
            # take the split file and load into video_list all the train/validation samples (by reading the split file)
            for i, _ in self.split_file.iterrows():
                index = self.split_file.loc[i, 'index']
                filename = self.split_file.loc[i, 'file']
                agent = filename[:5]
                print(index)
                print(filename)
                video = self.samples_dict[agent].loc[index, ['start_frame', 'stop_frame', 'description', 'label']]
                self.video_list.append(ActionRecord(video))
        else:
            # load the already extracted features to be the RGB samples
            # TODO: load features for all agents (instead of D1 use S00_2, S01_1, ...)
            self.model_features['RGB'] = pd.DataFrame(pd.read_pickle(f'saved_features/action_net/{sampling}_{n_frames}_D1_{mode}')['features'])['features_RGB']
            
            # load into EMG mode the samples of the corresponding split
            # load into specto mode the samples of the corresponding split
            for i, _ in self.split_file.iterrows():
                index = self.split_file.loc[i, 'index']
                filename = self.split_file.loc[i, 'file']
                agent = filename[:5]
                self.model_features['EMG'] = self.model_features.append(self.samples_dict[agent].loc[index, [
                                                                                                    'start', 
                                                                                                    'stop', 
                                                                                                    'myo_left_timestamps', 
                                                                                                    'myo_left_readings', 
                                                                                                    'myo_right_timestamps', 
                                                                                                    'myo_right_readings', 
                                                                                                    'description', 
                                                                                                    'label']], ignore_index=True)
                self.model_features['specto'] = self.model_features.append(self.samples_dict[agent].loc[index, [
                                                                                                    'file', 
                                                                                                    'description', 
                                                                                                    'label']], ignore_index=True)
                                                                               
    def _get_train_indices(self, record: ActionRecord, modality='RGB'):
        start_frame = 0
        end_frame = record.num_frames[modality]
        
        frames_per_clip = self.num_frames_per_clip[modality]
        
        selected_frames = []
        
        for _ in range(self.num_clips):
            # If the number of frames of the clip is not sufficient
            # the remaining ones are chosen randomly from the sequence
            clip_frames = []
            if end_frame < frames_per_clip:
                clip_frames = list(range(start_frame, end_frame))
                
                while len(clip_frames) < frames_per_clip:
                    clip_frames.append(random.randint(start_frame, end_frame))
                
                clip_frames.sort()
            
            else:
                if self.dense_sampling.get(modality, False):
                    step = self.stride + 1
                    # frames_per_side = (self.num_frames_per_clip[modality]-1)//2
                    dead_select_zone = frames_per_clip * (step)
                    
                    clip_start_frame = random.randint(start_frame, max(start_frame, end_frame-dead_select_zone))
                    
                    clip_frames = list(
                        range(
                            clip_start_frame, 
                            min(end_frame, clip_start_frame+dead_select_zone),
                            step
                            )
                        )
                    
                    if len(clip_frames) < frames_per_clip:
                        available_frames = list(i for i in range(start_frame, end_frame+1) if i not in clip_frames)
                        
                        while len(clip_frames) < frames_per_clip:
                            sel = random.choice(available_frames)
                            clip_frames.append(sel)
                            available_frames.remove(sel)
                        
                        clip_frames.sort()
                else:
                    higher_bound = end_frame//frames_per_clip
                    step = max(1, random.randint(higher_bound//2, higher_bound))
                    
                    clip_start_frame = random.randint(start_frame, end_frame-step*(frames_per_clip-1))
                    clip_end_frame = clip_start_frame + step * frames_per_clip
                    clip_frames = list(range(clip_start_frame, clip_end_frame, step))
            
            selected_frames.append(clip_frames)
        
        to_return = []
        selected_frames.sort(key=lambda i: i[0])
        for clip in selected_frames:
            to_return.extend(clip)
        
        return to_return

    def _get_val_indices(self, record: ActionRecord, modality='RGB'):
        start_frame = 0
        end_frame = record.num_frames[modality]
        
        frames_per_clip = self.num_frames_per_clip[modality]
        
        selected_frames = []
        
        for _ in range(self.num_clips):
            # If the number of frames of the clip is not sufficient
            # the remaining ones are chosen randomly from the sequence
            clip_frames = []
            if end_frame < frames_per_clip:
                clip_frames = list(range(start_frame, end_frame))
                
                while len(clip_frames) < frames_per_clip:
                    clip_frames.append(random.randint(start_frame, end_frame))
                
                clip_frames.sort()
            
            else:
                if self.dense_sampling.get(modality, False):
                    step = self.stride + 1
                    # frames_per_side = (self.num_frames_per_clip[modality]-1)//2
                    dead_select_zone = frames_per_clip * (step)
                    
                    clip_start_frame = random.randint(start_frame, max(start_frame, end_frame-dead_select_zone))
                    
                    clip_frames = list(
                        range(
                            clip_start_frame, 
                            min(end_frame, clip_start_frame+dead_select_zone),
                            step
                            )
                        )
                    
                    if len(clip_frames) < frames_per_clip:
                        available_frames = list(i for i in range(start_frame, end_frame+1) if i not in clip_frames)
                        
                        while len(clip_frames) < frames_per_clip:
                            sel = random.choice(available_frames)
                            clip_frames.append(sel)
                            available_frames.remove(sel)
                        
                        clip_frames.sort()
                else:
                    higher_bound = end_frame//frames_per_clip
                    step = max(1, random.randint(higher_bound//2, higher_bound))
                    
                    clip_start_frame = random.randint(start_frame, end_frame-step*(frames_per_clip-1))
                    clip_end_frame = clip_start_frame + step * frames_per_clip
                    clip_frames = list(range(clip_start_frame, clip_end_frame, step))
            
            selected_frames.append(clip_frames)
        
        to_return = []
        selected_frames.sort(key=lambda i: i[0])
        for clip in selected_frames:
            to_return.extend(clip)
        
        return to_return

    def __getitem__(self, index):
        if self.extract_features:
            frames = {}
            segment_indices = {}
            record = self.video_list[index]
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)
            
            img, label = self.get('RGB', record, segment_indices[m])
            frames[m] = img

            return frames, label
        else:
            sample = {}
            for modality in self.modalities:
                sample[modality] = self.model_features[modality].loc[index, :]
            return sample, record.label
    
    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_data(modality, record, p)
            images.extend(frame)
        # finally, all the transformations are applied
        process_data = self.transform[modality](images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality == 'RGBDiff':
            # here the offset for the starting index of the sample is added

            idx_untrimmed = record.start_frame + idx
            try:
                img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed))) \
                    .convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                print(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed), idx, record.start_frame)
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(max_idx_video))) \
                        .convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]
        
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.data)
        

class EpicKitchensDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs):
        """
        split: str (D1, D2 or D3)
        modalities: list(str, str, ...)
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"

        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        self.video_list = [EpicVideoRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat

        if self.load_feat:
            self.model_features = None
            for m in self.modalities:
                # load features for each modality
                model_features = pd.DataFrame(pd.read_pickle(os.path.join("saved_features/epic_kitchen",
                                                                          self.dataset_conf[m].features_name + "_" +
                                                                          pickle_name))['features'])[["uid", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")

    def _get_train_indices(self, record: EpicVideoRecord, modality='RGB'):
        start_frame = 0
        end_frame = record.num_frames[modality]
        
        frames_per_clip = self.num_frames_per_clip[modality]
        
        selected_frames = []
        
        for _ in range(self.num_clips):
            # If the number of frames of the clip is not sufficient
            # the remaining ones are chosen randomly from the sequence
            clip_frames = []
            if end_frame < frames_per_clip:
                clip_frames = list(range(start_frame, end_frame))
                
                while len(clip_frames) < frames_per_clip:
                    clip_frames.append(random.randint(start_frame, end_frame))
                
                clip_frames.sort()
            
            else:
                if self.dense_sampling.get(modality, False):
                    step = self.stride + 1
                    # frames_per_side = (self.num_frames_per_clip[modality]-1)//2
                    dead_select_zone = frames_per_clip * (step)
                    
                    clip_start_frame = random.randint(start_frame, max(start_frame, end_frame-dead_select_zone))
                    
                    clip_frames = list(
                        range(
                            clip_start_frame, 
                            min(end_frame, clip_start_frame+dead_select_zone),
                            step
                            )
                        )
                    
                    if len(clip_frames) < frames_per_clip:
                        available_frames = list(i for i in range(start_frame, end_frame+1) if i not in clip_frames)
                        
                        while len(clip_frames) < frames_per_clip:
                            sel = random.choice(available_frames)
                            clip_frames.append(sel)
                            available_frames.remove(sel)
                        
                        clip_frames.sort()
                else:
                    higher_bound = end_frame//frames_per_clip
                    step = max(1, random.randint(higher_bound//2, higher_bound))
                    
                    clip_start_frame = random.randint(start_frame, end_frame-step*(frames_per_clip-1))
                    clip_end_frame = clip_start_frame + step * frames_per_clip
                    clip_frames = list(range(clip_start_frame, clip_end_frame, step))
            
            selected_frames.append(clip_frames)
        
        to_return = []
        selected_frames.sort(key=lambda i: i[0])
        for clip in selected_frames:
            to_return.extend(clip)
        
        return to_return

    def _get_val_indices(self, record: EpicVideoRecord, modality: 'RGB'):
        start_frame = 0
        end_frame = record.num_frames[modality]
        
        frames_per_clip = self.num_frames_per_clip[modality]
        
        selected_frames = []
        
        for _ in range(self.num_clips):
            # If the number of frames of the clip is not sufficient
            # the remaining ones are chosen randomly from the sequence
            clip_frames = []
            if end_frame < frames_per_clip:
                clip_frames = list(range(start_frame, end_frame))
                
                while len(clip_frames) < frames_per_clip:
                    clip_frames.append(random.randint(start_frame, end_frame))
                
                clip_frames.sort()
            
            else:
                if self.dense_sampling.get(modality, False):
                    step = self.stride + 1
                    # frames_per_side = (self.num_frames_per_clip[modality]-1)//2
                    dead_select_zone = frames_per_clip * (step)
                    
                    clip_start_frame = random.randint(start_frame, max(start_frame, end_frame-dead_select_zone))
                    
                    clip_frames = list(
                        range(
                            clip_start_frame, 
                            min(end_frame, clip_start_frame+dead_select_zone),
                            step
                            )
                        )
                    
                    if len(clip_frames) < frames_per_clip:
                        available_frames = list(i for i in range(start_frame, end_frame+1) if i not in clip_frames)
                        
                        while len(clip_frames) < frames_per_clip:
                            sel = random.choice(available_frames)
                            clip_frames.append(sel)
                            available_frames.remove(sel)
                        
                        clip_frames.sort()
                else:
                    higher_bound = end_frame//frames_per_clip
                    step = max(1, random.randint(higher_bound//2, higher_bound))
                    
                    clip_start_frame = random.randint(start_frame, end_frame-step*(frames_per_clip-1))
                    clip_end_frame = clip_start_frame + step * frames_per_clip
                    clip_frames = list(range(clip_start_frame, clip_end_frame, step))
            
            selected_frames.append(clip_frames)
        
        to_return = []
        selected_frames.sort(key=lambda i: i[0])
        for clip in selected_frames:
            to_return.extend(clip)
        
        return to_return

    def __getitem__(self, index):

        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]

        if self.load_feat:
            sample = {}
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
            assert len(sample_row) == 1
            for m in self.modalities:
                sample[m] = sample_row["features_" + m].values[0]
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_data(modality, record, p)
            images.extend(frame)
        # finally, all the transformations are applied
        process_data = self.transform[modality](images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality == 'RGBDiff':
            # here the offset for the starting index of the sample is added

            idx_untrimmed = record.start_frame + idx
            try:
                img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed))) \
                    .convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                print(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed), idx, record.start_frame)
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(max_idx_video))) \
                        .convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]
        
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.video_list)
