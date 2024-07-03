emg_descriptions_to_labels = [
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

class ActionRecord():
    def __init__(self, video):
        self._sample = video
        # video = ['start_frame', 'stop_frame', 'description', 'verb_class']
    
    @property
    def start_frame(self):
        return int(self._sample['start_frame'])

    @property
    def stop_frame(self):
        return int(self._sample['stop_frame'])

    @property
    def description(self):
        return self._sample['description']

    @property
    def label(self):
        return self._sample['label']

    @property
    def num_frames(self):
        return self.stop_frame - self.start_frame
