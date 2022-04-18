#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:05:03 2017
@author: hxj
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


num_epoch = 3000
num_classes = 4
batch_size = 128
is_adam = True
learning_rate = 0.001
dropout_keep_prob = 1
image_height = 300
image_width = 40
image_channel = 3
# traindata_path = './data/IEMOCAP.pkl'
# validdata_path = 'inputs/valid.pkl'
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
    train_data, train_label, test_data, test_label, _, _, _, Test_label, pernums_test, _ = load_data('./data/IEMOCAP.pkl')

    valid_data = np.load('data/IEMOCAP_valid_data.npy', allow_pickle=True)
    valid_label = np.load('data/IEMOCAP_valid_label.npy', allow_pickle=True)
    Valid_label = np.load('data/IEMOCAP_Valid_label.npy', allow_pickle=True)
    pernums_valid = np.load('data/IEMOCAP_pernums_valid.npy', allow_pickle=True)

    train_label = train_label.reshape(-1)
    valid_label = valid_label.reshape(-1)
    Valid_label = Valid_label.reshape(-1)

    valid_size = valid_data.shape[0]
    dataset_size = train_data.shape[0]
    vnum = pernums_valid.shape[0]
    best_valid_uw = 0
    device = 'cuda'

    ##########tarin model###########

    model = ACRNN()
    # model.apply(init_weights)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, threshold=0.02, factor=0.5, min_lr=1e-8)
    criterion = torch.nn.CrossEntropyLoss()

    # print(train_data.shape)        # (1200, 300, 40, 3)  # (B, H, W, C)
    train_data = train_data.transpose((0, 3, 1, 2))
    test_data = test_data.transpose((0, 3 ,1 ,2))
    valid_data = valid_data.transpose((0, 3 ,1 ,2))
    # print(train_data.shape)        # (1200, 3, 300, 40)  # (B, C, H, W)
    
    num_epoch = 3000
    train_iter = divmod(dataset_size, batch_size)[0]

    for epoch in range(num_epoch):
        # training
        model.train()
        shuffle_index = list(range(len(train_data)))
        np.random.shuffle(shuffle_index)
        running_loss = 0
        batch_bar = tqdm(total=train_iter, dynamic_ncols=True, leave=False)
        for i in range(train_iter):
            start = (i*batch_size) % dataset_size
            end = min(start+batch_size, dataset_size)

            if i == (train_iter-1) and end < dataset_size:
                end = dataset_size
        
            inputs = torch.tensor(train_data[shuffle_index[start:end]]).to(device)
            targets = torch.tensor(train_label[shuffle_index[start:end]], dtype=torch.long).to(device)
            optimizer.zero_grad()
            outputs, rnn_hs = model(inputs)
            loss = criterion(outputs, targets) + sum(ar_alpha * rnn_h.pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss.backward()
            running_loss += float(loss)
            if clip:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
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
             valid_iter = divmod(valid_size, batch_size)[0]
             y_pred_valid = np.empty((valid_size, num_classes),dtype=np.float32)
             y_valid = np.empty((vnum, 4), dtype=np.float32)
             index = 0     
             cost_valid = 0
             
             if (valid_size < batch_size):

                 # inference
                 with torch.no_grad():
                     inputs = torch.tensor(valid_data[v_begin:v_end]).to(device)
                     targets = torch.tensor(Valid_label[v_begin:v_end], dtype=torch.long).to(device)
                     outputs, _ = model(inputs)
                     y_pred_valid[v_begin:v_end,:] = outputs.cpu().detach().numpy()
                     loss = criterion(outputs, targets).cpu().detach().numpy()

                 cost_valid = cost_valid + np.sum(loss)
             
             for v in range(valid_iter):
                 v_begin, v_end = v*batch_size, (v+1)*batch_size

                 if v == (valid_iter-1) and v_end < valid_size:
                     v_end = valid_size

                 # inference
                 with torch.no_grad():
                     inputs = torch.tensor(valid_data[v_begin:v_end]).to(device)
                     targets = torch.tensor(Valid_label[v_begin:v_end], dtype=torch.long).to(device)
                     outputs, _ = model(inputs)
                     y_pred_valid[v_begin:v_end,:] = outputs.cpu().detach().numpy()
                     loss = criterion(outputs, targets).cpu().detach().numpy()
                  
                 cost_valid = cost_valid + np.sum(loss)

             cost_valid = cost_valid/valid_size

             for s in range(vnum):
                 y_valid[s,:] = np.max(y_pred_valid[index:index+pernums_valid[s],:], 0)
                 index = index + pernums_valid[s]
                 
             # compute evaluated results
             valid_acc_uw = recall(valid_label, np.argmax(y_valid, 1), average='macro')
             valid_conf = confusion(valid_label, np.argmax(y_valid, 1))

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

        if valid_acc_uw >= 0.3:
            scheduler.step(valid_acc_uw)

def test():
    pass


if __name__=='__main__':
    train()
