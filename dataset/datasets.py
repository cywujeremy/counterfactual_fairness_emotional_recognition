import sys
sys.path.append('..')

import os
import csv
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F

from data.emotion_labels import EMOTIONAL_LABELS, EMOTIONAL_LABELS_SHORT

class IEMOCAPSamples(torch.utils.data.Dataset):
    
    def __init__(self, data_path, partition='train', shuffle=True, padded_len=1500):
        super().__init__()
        
        self.data_dir = os.path.join(data_path, partition)
        self.data_files = os.listdir(self.data_dir)
        self.padded_len = padded_len
        
        if shuffle == True:
            random.shuffle(self.data_files)
        
    def __len__(self):

        return len(self.data_files)

    def __getitem__(self, idx):

        data_path = self.data_dir + "\\" + self.data_files[idx]
        label, features = np.load(data_path, allow_pickle=True)
        
        features = np.pad(features, ((0, self.padded_len - features.shape[0]), (0, 0), (0, 0)))
        label = EMOTIONAL_LABELS.index(label)
        
        return features, label
        

class IEMOCAPSmall(torch.utils.data.Dataset):

    def __init__(self, pickle_path="data/IEMOCAP.pkl", partition='train'):
        super().__init__()

        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        
        if partition == "train":
            self.X = data[0]
            self.y = data[1].reshape(-1)
        
        elif partition == "dev":
            self.X = data[4]
            self.y = data[6].reshape(-1)
        
        elif partition == "test":
            self.X = data[2]
            self.y = data[7].reshape(-1)
        
    def __len__(self):

        return len(self.X)
    
    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]
        


        