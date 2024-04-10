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
    def __init__(self, tup):
        self._index = tup[0]
        self._sample = tup[1]
        # tup[1] = ['index', 'emg_file', 'rgb_file', 'spectograms_file', 'description', 'labels']

    @property
    def index(self):
        return self._sample['index']
    
    @property
    def emg_file(self):
        return self._sample['emg_file']

    @property
    def rgb_file(self):
        return self._sample['rgb_file']

    @property
    def spectograms_file(self):
        return self._sample['spectograms_file']

    @property
    def label(self):
        return emg_descriptions_to_labels.index(self._sample['description'])
