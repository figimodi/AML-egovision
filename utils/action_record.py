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

class ActionRecord():
    def __init__(self, video):
        self._index = video[0]
        self._sample = video[1]
        # video[1] = ['start_frame', 'stop_frame', 'description', 'verb_class']
    
    @property
    def start_frame(self):
        return self._sample['start_frame']

    @property
    def stop_frame(self):
        return self._sample['stop_frame']

    @property
    def description(self):
        return self._sample['description']

    @property
    def label(self):
        return self._sample['label']

    @property
    def num_frames(self):
        return {'RGB': self.stop_frame - self.start_frame}
