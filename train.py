"""
This training pipeline is adapted from the codes of the original paper.
Reference link: https://github.com/Chien-Hung/Speech-Emotion-Recognition
"""

from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import numpy as np
from model import ACRNN
import pickle
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion
import os
import torch
import torch.optim as optim
import pdb
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import IEMOCAPTrain, IEMOCAPValid


num_epoch = 3000
num_classes = 4
batch_size = 128
is_adam = True
learning_rate = 0.001
dropout_keep_prob = 1
image_height = 300
image_width = 40
image_channel = 3

checkpoint = './checkpoint'
model_name = 'best_model.pth'
clip = 0
ar_alpha = 0.2

def load_data(in_dir):
    f = open(in_dir,'rb')
    train_data,train_label,test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid = pickle.load(f)
    #train_data,train_label,test_data,test_label,valid_data,valid_label = pickle.load(f)
    return train_data,train_label,test_data,test_label,valid_data,valid_label,Valid_label,Test_label,pernums_test,pernums_valid

def train():
    #####load data##########
    best_valid_uw = 0
    device = 'cuda'

    train_dataset = IEMOCAPTrain()
    valid_dataset = IEMOCAPValid()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    vnum = valid_dataset.pernums_valid.shape[0]
    ##########tarin model###########

    model = ACRNN()
    # model.apply(init_weights)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=15, threshold=0.02, factor=0.5, min_lr=1e-8)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        # training
        model.train()
        running_loss = 0
        batch_bar = tqdm(total=(len(train_loader)), dynamic_ncols=True, leave=False)
        for i, (inputs, targets) in enumerate(train_loader):
            
            inputs, targets = inputs.float().cuda(), targets.long().cuda()

            optimizer.zero_grad()
            outputs, rnn_hs = model(inputs)
            loss = criterion(outputs, targets) + sum(ar_alpha * rnn_h.pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss.backward()
            running_loss += float(loss)

            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            batch_bar.set_postfix(
                loss=f"{float(running_loss / (i + 1)):.4f}",
                lr=f"{float(optimizer.param_groups[0]['lr']):.6f}"
            )
            batch_bar.update()

        batch_bar.close()
        print(f"Epoch {epoch}/{num_epoch}: loss={running_loss / (i + 1):.4f}")
        
        if epoch % 1 == 0:
            # validation
            model.eval()
            y_pred_valid = np.empty((0, num_classes),dtype=np.float32)
            y_valid = np.empty((vnum, 4), dtype=np.float32)
            index = 0     
            cost_valid = 0
             
            for i, (inputs, targets) in enumerate(valid_loader):
                with torch.no_grad():
                    inputs, targets = inputs.float().cuda(), targets.long().cuda()
                    outputs, _ = model(inputs)
                    y_pred_valid = np.vstack((y_pred_valid, outputs.detach().cpu().numpy()))
                    loss = criterion(outputs, targets).cpu().detach().numpy()
                cost_valid = cost_valid + np.sum(loss)

            cost_valid = cost_valid / len(valid_dataset)

            for s in range(vnum):
                y_valid[s,:] = np.max(y_pred_valid[index:index + valid_dataset.pernums_valid[s],:], 0)
                index = index + valid_dataset.pernums_valid[s]
                 
             # compute evaluated results
            valid_acc_uw = recall(valid_dataset.valid_label, np.argmax(y_valid, 1), average='macro')
            valid_conf = confusion(valid_dataset.valid_label, np.argmax(y_valid, 1))

             # save the best val result
            if valid_acc_uw > best_valid_uw:
                best_valid_uw = valid_acc_uw
                best_valid_conf = valid_conf

                if not os.path.isdir(checkpoint):
                    os.mkdir(checkpoint)
                torch.save(model.state_dict(), os.path.join(checkpoint, model_name))

            # print results
            print ("*****************************************************************")
            print ("Epoch: %05d" %(epoch+1))
            # print ("Training cost: %2.3g" %tcost)   
            # print ("Training accuracy: %3.4g" %tracc) 
            print ("Valid cost: %2.3g" %cost_valid)
            print ("Valid_UA: %3.4g" %valid_acc_uw)    
            print ("Best valid_UA: %3.4g" %best_valid_uw) 
            print ('Valid Confusion Matrix:["ang","sad","hap","neu"]')
            print (valid_conf)
            print ('Best Valid Confusion Matrix:["ang","sad","hap","neu"]')
            print (best_valid_conf)
            print (f"Learning Rate: {optimizer.param_groups[0]['lr']}")
            print ("*****************************************************************")

        if valid_acc_uw >= 0.3 and epoch >= 20:
            scheduler.step(valid_acc_uw)

def test():
    pass


if __name__=='__main__':
    train()
