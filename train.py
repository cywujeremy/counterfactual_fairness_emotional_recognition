"""
This training pipeline is adapted from the codes of the original paper.
Reference link: https://github.com/Chien-Hung/Speech-Emotion-Recognition
"""

from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import numpy as np
import time
from model import ACRNN
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion
import os
import torch
import torch.optim as optim
import pdb
from tqdm import tqdm

from torch.utils.data import DataLoader
from datasets import IEMOCAPTrain, IEMOCAPValid
from utils.training_tracker import TrainingTracker




def train():

    num_epoch = 300
    num_classes = 4
    batch_size = 128
    learning_rate = 0.001

    start_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    checkpoint = f'./checkpoint/{start_time}'
    experiment_name = "acrnn_locked_dropout_act_reg0.3"

    clip = 0
    ar_alpha = 0.3

    best_valid_uw = 0
    device = 'cuda'

    tracker = TrainingTracker(experiment_name)

    train_dataset = IEMOCAPTrain()
    valid_dataset = IEMOCAPValid()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    vnum = valid_dataset.pernums_valid.shape[0]

    model = ACRNN()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=15, threshold=0.02, factor=0.8, min_lr=1e-8)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0
        batch_bar = tqdm(total=(len(train_loader)), dynamic_ncols=True, leave=False)

        y_pred = torch.tensor([])
        y_true = torch.tensor([])

        for i, (inputs, targets) in enumerate(train_loader):
            
            inputs, targets = inputs.float().cuda(), targets.long().cuda()
            optimizer.zero_grad()
            outputs, rnn_hs = model(inputs)

            y_true = torch.cat((y_true, targets.cpu()))
            y_pred = torch.cat((y_pred, outputs.argmax(dim=1).cpu()))

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

        tracker.training_loss.append(running_loss / (i + 1))
        tracker.training_accuracy.append(accuracy_score(y_true, y_pred))
        tracker.training_uar.append(recall(y_true, y_pred, average='macro'))

        batch_bar.close()
        print(f"Epoch {epoch + 1}/{num_epoch}: loss={running_loss / (i + 1):.4f}")
        
        if epoch % 1 == 0:
            # validation
            model.eval()
            y_pred_valid = np.empty((0, num_classes), dtype=np.float32)
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

            cost_valid = cost_valid / len(valid_loader)

            for s in range(vnum):
                y_valid[s,:] = np.max(y_pred_valid[index:index + valid_dataset.pernums_valid[s],:], 0)
                index = index + valid_dataset.pernums_valid[s]
                 
            # compute evaluated results
            valid_accuracy = accuracy_score(valid_dataset.valid_label, np.argmax(y_valid, 1))
            valid_acc_uw = recall(valid_dataset.valid_label, np.argmax(y_valid, 1), average='macro')
            valid_conf = confusion(valid_dataset.valid_label, np.argmax(y_valid, 1))
            tracker.validation_accuracy.append(valid_accuracy)
            tracker.validation_uar.append(valid_acc_uw)
            tracker.validation_confusion_matrix.append(valid_conf)
            tracker.validation_loss.append(cost_valid)

             # save the best val result
            if valid_acc_uw > best_valid_uw:
                best_valid_uw = valid_acc_uw
                best_valid_conf = valid_conf
                tracker.best_epoch = epoch

                if not os.path.isdir(checkpoint):
                    os.mkdir(checkpoint)

                model_name = f"model_{experiment_name}_{best_valid_uw:.4f}.pth"
                torch.save(model.state_dict(), os.path.join(checkpoint, model_name))

            # print results
            print ("*****************************************************************")
            print ("Epoch: %05d" %(epoch + 1))
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
        
        log_name = f"log/log_{start_time}_{experiment_name}.pkl"
        with open(log_name, 'wb') as f:
            pickle.dump(tracker, f)

def test():
    pass


if __name__=='__main__':
    train()
