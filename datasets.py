import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import random

def load_data(in_dir):
    with open(in_dir, 'rb') as f:
        train_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label, pernums_test, pernums_valid = pickle.load(f)
    return train_data, train_label, test_data, test_label, valid_data, valid_label, Valid_label, Test_label, pernums_test, pernums_valid

class IEMOCAPTrain(Dataset):

    def __init__(self, root='data', converted=False):
        super(IEMOCAPTrain, self).__init__()
        path = root + '/IEMOCAP_train.pkl' if converted == False else root + '/IEMOCAP_train_converted.pkl'
        data = load_data(path)
        self.train_data, self.train_label = data[0].transpose((0, 3, 1, 2)), data[1]
        self.train_label = self.train_label.reshape(-1)

    def __getitem__(self, index):
        return torch.tensor(self.train_data[index]), torch.tensor(self.train_label[index])

    def __len__(self):
        return len(self.train_data)
        
class IEMOCAPTrainAUG(Dataset):

    def __init__(self, root='data', ratio=1.0):
        super(IEMOCAPTrainAUG, self).__init__()
        orgdata = load_data(root + '/IEMOCAP_train.pkl')
        counterdata = load_data(root + '/IEMOCAP_train_converted.pkl')

        # select the subset of dataset based on a ratio
        randnum = random.sample(range(0, orgdata[0].shape[0]), int(orgdata[0].shape[0]*ratio))
        self.train_data = np.concatenate([orgdata[0], counterdata[0][randnum]]).transpose((0, 3, 1, 2))
        self.train_label = np.concatenate([orgdata[1], counterdata[1][randnum]])
        self.train_label = self.train_label.reshape(-1)
    
    def __getitem__(self, index):
        return torch.tensor(self.train_data[index]), torch.tensor(self.train_label[index])

    def __len__(self):
        return len(self.train_data)


class IEMOCAPEval(Dataset):

    def __init__(self, root='data', partition='val'):
        
        super(IEMOCAPEval, self).__init__()

        if partition == 'val':
            data_files={'valid_data': '/IEMOCAP_valid_data.npy',
                        'valid_label': '/IEMOCAP_valid_label.npy',
                        'valid_gender': '/IEMOCAP_valid_gender.npy'}

            self.data = np.load(root + data_files['valid_data'], allow_pickle=True).transpose((0, 3 ,1 ,2))
            self.label = np.load(root + data_files['valid_label'], allow_pickle=True).reshape(-1)
            self.gender = np.load(root + data_files['valid_gender'], allow_pickle=True).reshape(-1)
            
        elif partition == 'test':
            data_files={'test_data': '/IEMOCAP_test_data.npy',
                        'test_label': '/IEMOCAP_test_label.npy',
                        'test_gender': '/IEMOCAP_test_gender.npy'}
            
            self.data = np.load(root + data_files['test_data'], allow_pickle=True).transpose((0, 3 ,1 ,2))
            self.label = np.load(root + data_files['test_label'], allow_pickle=True).reshape(-1)
            self.gender = np.load(root + data_files['test_gender'], allow_pickle=True).reshape(-1)
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index]), torch.tensor(self.label[index])

    def __len__(self):
        return len(self.data)