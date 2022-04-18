import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

def load_data(in_dir):
    with open(in_dir, 'rb') as f:
        train_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label, pernums_test, pernums_valid = pickle.load(f)
    return train_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label, pernums_test, pernums_valid

class IEMOCAPTrain(Dataset):

    def __init__(self, root='data'):
        super(IEMOCAPTrain, self).__init__()
        path = root + '/IEMOCAP.pkl'
        data = load_data(path)
        self.train_data, self.train_label = data[0].transpose((0, 3, 1, 2)), data[1]
        self.train_label = self.train_label.reshape(-1)
    
    def __getitem__(self, index):
        return torch.tensor(self.train_data[index]), torch.tensor(self.train_label[index])

    def __len__(self):
        return len(self.train_data)

class IEMOCAPValid(Dataset):

    def __init__(self, root='data', 
                 data_files={'valid_data': '/IEMOCAP_valid_data.npy',
                             'valid_label': '/IEMOCAP_valid_label.npy',
                             'Valid_label': '/IEMOCAP_Valid_label.npy',
                             'pernums_valid': '/IEMOCAP_pernums_valid.npy'}):
        
        super(IEMOCAPValid, self).__init__()
        self.valid_data = np.load(root + data_files['valid_data'], allow_pickle=True).transpose((0, 3 ,1 ,2))
        self.valid_label = np.load(root + data_files['valid_label'], allow_pickle=True).reshape(-1)
        self.Valid_label = np.load(root + data_files['Valid_label'], allow_pickle=True).reshape(-1)
        self.pernums_valid = np.load(root + data_files['pernums_valid'], allow_pickle=True)
    
    def __getitem__(self, index):
        return torch.tensor(self.valid_data[index]), torch.tensor(self.Valid_label[index])

    def __len__(self):
        return len(self.valid_data)