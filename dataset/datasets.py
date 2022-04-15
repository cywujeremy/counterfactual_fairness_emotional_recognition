import sys
sys.path.append('..')

import os
import csv
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from data.emotion_labels import EMOTIONAL_LABELS, EMOTIONAL_LABELS_SHORT

class IEMOCAPSamples(torch.utils.data.Dataset):
    
    def __init__(self, data_path, partition='train', shuffle=True):
        super().__init__()
        
        self.data_dir = os.path.join(data_path, partition)
        self.data_files = os.listdir(self.data_dir)
        
        if shuffle == True:
            random.shuffle(self.data_files)
        
    def __len__(self):

        return len(self.data_files)

    def __getitem__(self, idx):

        data_path = self.data_dir + "/" + self.data_files[idx]
        label, features = np.load(data_path, allow_pickle=True)
        label = EMOTIONAL_LABELS.index(label)
        
        return features, label
    
    @staticmethod
    def collate_fn(batch):

        batch_x = [torch.tensor(x) for x, _ in batch]
        batch_y = [torch.tensor(y) for _, y in batch]

        batch_x_pad = pad_sequence(batch_x)
        lengths_x = [len(x) for x in batch_x]

        return batch_x_pad[:600], torch.tensor(lengths_x), torch.tensor(batch_y) 


        

# class IEMOCAPSmall(torch.utils.data.Dataset):

#     def __init__(self, pickle_path="data/IEMOCAP.pkl", partition='train'):
#         super().__init__()

#         with open(pickle_path, "rb") as f:
#             data = pickle.load(f)
        
#         if partition == "train":
#             self.X = data[0]
#             self.y = data[1].reshape(-1)
        
#         elif partition == "dev":
#             self.X = data[4]
#             self.y = data[6].reshape(-1)
        
#         elif partition == "test":
#             self.X = data[2]
#             self.y = data[7].reshape(-1)
        
#     def __len__(self):

#         return len(self.X)
    
#     def __getitem__(self, idx):

#         return self.X[idx], self.y[idx]

class IEMOCAPSmall(torch.utils.data.Dataset):

    def __init__(self, pickle_path="data/IEMOCAP.pkl", partition='train', dev_samples=428):
        super().__init__()
        
        dev_sample_range = dev_samples

        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        
        if partition == "train":
            self.X = data[0]
            self.y = data[1].reshape(-1)
        
        elif partition == "dev":
            self.X = data[2][:dev_sample_range]
            self.y = data[1][:dev_sample_range]
        
        elif partition == "test":
            self.X = data[2][dev_sample_range:]
            self.y = data[1][dev_sample_range:]
        
    def __len__(self):

        return len(self.X)
    
    def __getitem__(self, idx):

        # X_normalized = ( - self.X[idx].mean(axis=1)) / self.X[idx].std(axis=1)

        return self.X[idx], self.y[idx]
        


        