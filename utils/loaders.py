import glob
from abc import ABC
import random
from emg_extract import emg_adjust_features
import pandas as pd
from .epic_record import EpicVideoRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
from utils.logger import logger

import numpy as np

# Clear cutting board                                                                           8
# Clean a plate with a towel                                                                    5
# Pour water from a pitcher into a glass                                                        5
# Get/replace items from refrigerator/cabinets/drawers                                          4
# Peel a cucumber                                                                               3
# Slice bread                                                                                   3
# Clean a plate with a sponge                                                                   3
# Open/close a jar of almond butter                                                             3
# Spread jelly on a bread slice                                                                 3
# Clean a pan with a towel                                                                      3
# Spread almond butter on a bread slice                                                         3
# Peel a potato                                                                                 3
# Slice a potato                                                                                3
# Slice a cucumber                                                                              2
# Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils    2
# Get items from refrigerator/cabinets/drawers                                                  2
# Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils            1
# Stack on table: 3 each large/small plates, bowls                                              1
# Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils                  1
# Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils          1

emg_descriptions_to_labels = [
    'Clear cutting board',
    'Clean a plate with a towel',
    'Pour water from a pitcher into a glass',
    'Get/replace items from refrigerator/cabinets/drawers',
    'Peel a cucumber',
    'Slice bread',
    'Clean a plate with a sponge',
    'Open/close a jar of almond butter',
    'Spread jelly on a bread slice',
    'Clean a pan with a towel',
    'Spread almond butter on a bread slice',
    'Peel a potato',
    'Slice a potato',
    'Slice a cucumber',
    'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
    'Get items from refrigerator/cabinets/drawers',
    'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
    'Stack on table: 3 each large/small plates, bowls',
    'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
    'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
]

class EmgDataset(data.Dataset, ABC):
    def __init__(self, data_path, index_file_split) -> None:
        file_name = f'./action-net/ActionNet_{index_file_split}.pkl'
        self.indices = pd.DataFrame(pd.read_pickle(file_name))
        
        temp_dict = {
            'description': [],
            'myo_left_readings': [],
            'myo_right_readings': [],
            # '': [],
            # '': [],
        }
        
        readings = {}
        
        for i, row in self.indices.iterrows():
            if i == 0:
                continue
            
            file = row['file']
            if file not in readings:
                readings[file] = emg_adjust_features(os.path.join(data_path, row['file']))
            
            for k in temp_dict.keys():
                temp_dict[k].append(readings[file][k][row['index']])

        self.data = pd.DataFrame(temp_dict)
    
    def __getitem__(self, index):
        element = self.data.loc[index, 'myo_left_readings':'myo_right_readings']
        label = emg_descriptions_to_labels.index(self.data.loc[index, 'description'])
        
        return element, label
    
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
                model_features = pd.DataFrame(pd.read_pickle(os.path.join("saved_features",
                                                                          self.dataset_conf[m].features_name + "_" +
                                                                          pickle_name))['features'])[["uid", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")


    def _get_train_indices(self, record, modality='RGB'):
        start_frame = 0
        end_frame = record.num_frames[modality]
        frames_per_clip = self.num_frames_per_clip[modality]
        tot_num_frames = frames_per_clip*self.num_clips
        selected_frames = []
        
        for nc in range(self.num_clips):
            random.seed(nc + self.num_clips)
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
                        available_frames = list(i for i in range(max(start_frame, central_frame - frames_select_zone), min(end_frame, central_frame + frames_select_zone)+1) if i not in clip_frames)
                        
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
        
        return selected_frames

    def _get_val_indices(self, record: EpicVideoRecord, modality):
        start_frame = 0
        end_frame = record.num_frames[modality]
        frames_per_clip = self.num_frames_per_clip[modality]
        tot_num_frames = frames_per_clip*self.num_clips
        selected_frames = []
        
        for nc in range(self.num_clips):
            random.seed(nc + self.num_clips)
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
                        available_frames = list(i for i in range(max(start_frame, central_frame - frames_select_zone), min(end_frame, central_frame + frames_select_zone)+1) if i not in clip_frames)
                        
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
        
        return selected_frames

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
