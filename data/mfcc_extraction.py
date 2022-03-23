import os
import glob
import pickle
import wave
import numpy as np
import python_speech_features as psf


class IEMOCAP_Dataset:
    
    def __init__(self, root, output_root=None, validation="default"):
        
        self.root = root
        
        if not output_root:
            self.output_root = self.root
        else:
            self.output_root = output_root
        
        self.validation = validation
        self.sessions = [session_name for session_name in os.listdir(self.root) if session_name[0] == 'S']
        self.utterance_list, self.utterance_paths = self._summarize_utterance_names()
        self.label_map = self._make_label_map()
    
    def _summarize_utterance_names(self):
        """summarize all sample names into a list
        """
        utter_list = []
        utter_paths = []
        
        for session in self.sessions:
            wav_folders = os.listdir(os.path.join(self.root, session, 'sentences', 'wav'))
            for wav_folder in wav_folders:
                wav_paths = glob.glob(os.path.join(self.root, session, 'sentences', 'wav', wav_folder) + '\\*.wav')
                wav_names = [name[:-4] for name in os.listdir(os.path.join(self.root, session, 'sentences', 'wav', wav_folder)) if name[-4:] == '.wav']
                
                if len(wav_names) != len(wav_paths):
                    a = 0
                
                utter_list += wav_names
                utter_paths += wav_paths
        
        assert len(utter_list) == len(utter_paths)
        
        return utter_list, utter_paths
    
    def _make_label_map(self):
        """Parse .txt files and collect labels
        """
        # initialize label map
        label_map = {}
        
        # loop through text files in session folders to collect labels and add to self.label_map
        for session in self.sessions:
            emo_eval_path = os.path.join(self.root, session, 'dialog', 'EmoEvaluation')
            txts = glob.glob(emo_eval_path + '\\*.txt')
            
            for txt in txts:
                # Reference: Source code of Chen et al., 2018.
                with open(txt, 'r') as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        if line[0] == '[':
                            t = line.split()
                            label_map[t[3]] = t[4]
        return label_map
        
    def _gen_features(self, utter_path):
        """Generate features from .wav files
        """
        # TODO: extract MFCC and deltas from wav files
        raise NotImplementedError
        
        
    
    def _gen_labels(self, utter_name):
        """Generate labels from label map
        """
        return self.label_map[utter_name]

    def make_dataset(self):
        """save (label, feature) tuples to file system
        """
        # TODO: organize and output label and feature tuples to file system
        raise NotImplementedError
        
        
    
    
    
    
        