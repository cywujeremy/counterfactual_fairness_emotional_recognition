import os
import glob
import pickle
import wave
from tqdm import tqdm
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
        
    def _gen_features(self, utter_path, feat_type='logfbank', delta=2):
        """Generate features from .wav files
        """
        # TODO: extract MFCC/LogFBank and deltas from wav files
        with wave.open(utter_path, 'r') as f:
            
            nchannels, sampwidth, framerate, wav_length = f.getparams()[:4]
            data = np.frombuffer(f.readframes(wav_length), dtype=np.short)
            time = np.arange(0,wav_length) * (1.0/framerate)
        
        mel_spec = psf.logfbank(data, framerate, nfilt=40)
        delta1 = psf.delta(mel_spec, 2)
        delta2 = psf.delta(delta1, 2)
        
        return np.stack((mel_spec, delta1, delta2), axis=1)
            
    
    def _gen_labels(self, utter_name):
        """Generate labels from label map
        """
        return self.label_map[utter_name]

    def make_dataset(self, train_val_split='default'):
        """save (label, feature) tuples to file system
        """
        # TODO: organize and output label and feature tuples to file system
        
        train_output_path = os.path.join(self.output_root, 'train')
        dev_output_path = os.path.join(self.output_root, 'dev')
        test_output_path = os.path.join(self.output_root, 'test')
        
        if train_val_split == 'default':
            
            train_sessions = ['Ses01', 'Ses02', 'Ses03']
            dev_sessions = ['Ses04']
            test_sessions = ['Ses05']
            
            train_wav_idx = [i for i in range(len(self.utterance_list)) if self.utterance_list[i][:5] in train_sessions]
            dev_wav_idx = [i for i in range(len(self.utterance_list)) if self.utterance_list[i][:5] in dev_sessions]
            test_wav_idx = [i for i in range(len(self.utterance_list)) if self.utterance_list[i][:5] in test_sessions]
        
        for idx in tqdm(train_wav_idx, total=len(train_wav_idx)):
            label_data = np.array((self._gen_labels(self.utterance_list[idx]), self._gen_features(self.utterance_paths[idx])), dtype=object)
            np.save(os.path.join(train_output_path, self.utterance_list[idx] + '.npy'), label_data)
        
        for idx in tqdm(dev_wav_idx, total=len(dev_wav_idx)):
            label_data = np.array((self._gen_labels(self.utterance_list[idx]), self._gen_features(self.utterance_paths[idx])), dtype=object)
            np.save(os.path.join(dev_output_path, self.utterance_list[idx] + '.npy'), label_data)
            
        for idx in tqdm(test_wav_idx, total=len(test_wav_idx)):
            label_data = np.array((self._gen_labels(self.utterance_list[idx]), self._gen_features(self.utterance_paths[idx])), dtype=object)
            np.save(os.path.join(test_output_path, self.utterance_list[idx] + '.npy'), label_data)
        
    
    
    
    
if __name__ == '__main__':
    
    ROOT = 'E:\\Download\\IEMOCAP_full_release_withoutVideos\\IEMOCAP_full_release'
    OUTPUT_ROOT = 'E:\\Download\\IEMOCAP_full_release_withoutVideos\\data'
    
    iemocap = IEMOCAP_Dataset(ROOT, OUTPUT_ROOT, validation='default')
    iemocap.make_dataset()