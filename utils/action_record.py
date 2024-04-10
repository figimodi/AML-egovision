# Index(['old_index', 'description', 'start', 'stop', 'myo_left_timestamps',
#        'myo_left_readings', 'myo_right_timestamps', 'myo_right_readings'],
#       dtype='object')
# >>> f.columns
# Index(['uid', 'participant_id', 'video_id', 'narration', 'start_timestamp',
#        'stop_timestamp', 'start_frame', 'stop_frame', 'verb', 'verb_class'],
#       dtype='object')

class ActionRecord():
    def __init__(self, tup):
        self._index = tup[0]
        self._sample = tup[1]

    @property
    def index(self):
        return self._sample['index']

    @property
    def start_frame(self):
        return self._sample['start_frame']

    @property
    def end_frame(self):
        return self._sample['stop_frame']

    @property
    def start_timestamp(self):
        return self._sample['start']

    @property
    def stop_timestamp(self):
        return self._sample['stop']
    
    @property
    def num_frames(self):
        return {'RGB': self.end_frame - self.start_frame,
                'EMG': len(min(self._sample['myo_left_timestamps'], self._sample['myo_right_timestamps'])),
                'Specto': 1}

    @property
    def label(self):
        if 'verb_class' not in self._sample.keys().tolist():
            raise NotImplementedError
        return self._sample['verb_class']
