"""
This training pipeline is adapted and reconstructed from the codes of the original paper.
Reference link: https://github.com/Chien-Hung/Speech-Emotion-Recognition
"""

from __future__ import absolute_import
from __future__ import division
from IPython.display import clear_output

import numpy as np
from copy import deepcopy
import time
from model import ACRNN, ACRNN_LockedDropout_WGTInit, ACRNNBaselineStable
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion
import os
import torch
import torch.optim as optim
import pdb
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import IEMOCAPTrain, IEMOCAPEval, IEMOCAPTrainTwin
from utils.training_tracker import TrainingTracker
from utils.fairness_eval import FairnessEvaluation

num_epoch = 300
num_classes = 4
batch_size = 128
learning_rate = 0.001

start_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
checkpoint = f'./checkpoint/{start_time}'
experiment_name = "acrnn_locked_dropout_act_reg0.3"

clip = 0
ar_alpha = 0.3
twin_gamma = 0.2
device = 'cuda'

def train(model, experiment_name, num_epoch, verbose=True, save_model=True, return_best_model=False):

    best_valid_uw = 0
    best_model = None

    tracker = TrainingTracker(experiment_name)
    # train_dataset = IEMOCAPTrain()
    train_dataset = IEMOCAPTrainTwin()
    valid_dataset = IEMOCAPEval(partition='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
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

        for i, (inputs, targets, inputs_conv, _) in enumerate(train_loader):
            
            inputs, targets = inputs.float().cuda(), targets.long().cuda()
            inputs_conv = inputs_conv.float().cuda()
            optimizer.zero_grad()
            outputs, rnn_hs = model(inputs)

            y_true = torch.cat((y_true, targets.cpu()))
            y_pred = torch.cat((y_pred, outputs.argmax(dim=1).cpu()))

            with torch.no_grad():
                outputs_conv, rnn_hs_conv = model(inputs_conv)

            loss = criterion(outputs, targets) + (twin_gamma * criterion(outputs, outputs_conv)) + sum(ar_alpha * rnn_h.pow(2).mean() for rnn_h in rnn_hs[-1:])
            # loss = criterion(outputs, targets)

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
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epoch}: loss={running_loss / (i + 1):.4f}")
        
        if epoch % 1 == 0:
            # validation
            model.eval()
            y_pred_valid = np.empty((0, num_classes), dtype=np.float32)
            cost_valid = 0
             
            for i, (inputs, targets) in enumerate(valid_loader):
                with torch.no_grad():
                    inputs, targets = inputs.float().cuda(), targets.long().cuda()
                    outputs, _ = model(inputs)
                    y_pred_valid = np.vstack((y_pred_valid, outputs.detach().cpu().numpy()))
                    loss = criterion(outputs, targets).cpu().detach().numpy()
                cost_valid = cost_valid + np.sum(loss)

            cost_valid = cost_valid / len(valid_loader)
                 
            # compute evaluated results
            valid_accuracy = accuracy_score(valid_dataset.label, np.argmax(y_pred_valid, 1))
            valid_acc_uw = recall(valid_dataset.label, np.argmax(y_pred_valid, 1), average='macro')
            valid_conf = confusion(valid_dataset.label, np.argmax(y_pred_valid, 1))
            tracker.validation_accuracy.append(valid_accuracy)
            tracker.validation_uar.append(valid_acc_uw)
            tracker.validation_confusion_matrix.append(valid_conf)
            tracker.validation_loss.append(cost_valid)

             # save the best val result
            if valid_acc_uw > best_valid_uw:
                best_valid_uw = valid_acc_uw
                best_valid_conf = valid_conf
                tracker.best_epoch = epoch
                best_model = deepcopy(model)

                if not os.path.isdir(checkpoint):
                    os.mkdir(checkpoint)
                if save_model:
                    model_name = f"model_{experiment_name}.pth"
                    torch.save(model, os.path.join(checkpoint, model_name))

            # print results
            if verbose:
                print("*****************************************************************")
                print("Epoch: %05d" %(epoch + 1))
                print("Valid cost: %2.3g" %cost_valid)
                print("Valid_UAR: %3.4g" %valid_acc_uw)    
                print("Best valid_UA: %3.4g" %best_valid_uw) 
                print('Valid Confusion Matrix:["ang","sad","hap","neu"]')
                print(valid_conf)
                print('Best Valid Confusion Matrix:["ang","sad","hap","neu"]')
                print(best_valid_conf)
                print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
                print("*****************************************************************")

        if valid_acc_uw >= 0.3 and epoch >= 20:
            scheduler.step(valid_acc_uw)
        
        log_name = f"log/log_{start_time}_{experiment_name}.pkl"
        with open(log_name, 'wb') as f:
            pickle.dump(tracker, f)
    
    if return_best_model:
        return best_model


def test(model, test_loader, test_dataset, criterion, return_fairness_eval=False):
    # test
    model.eval()
    y_pred_test = np.empty((0, num_classes), dtype=np.float32)
    cost_test = 0
        
    for i, (inputs, targets) in enumerate(test_loader):
        with torch.no_grad():
            inputs, targets = inputs.float().cuda(), targets.long().cuda()
            outputs, _ = model(inputs)
            y_pred_test = np.vstack((y_pred_test, outputs.detach().cpu().numpy()))
            loss = criterion(outputs, targets).cpu().detach().numpy()
        cost_test = cost_test + np.sum(loss)

    cost_test = cost_test / len(test_loader)
            
    # compute performance evaluation scores
    test_accuracy = accuracy_score(test_dataset.label, np.argmax(y_pred_test, 1))
    test_acc_uw = recall(test_dataset.label, np.argmax(y_pred_test, 1), average='macro')
    test_conf = confusion(test_dataset.label, np.argmax(y_pred_test, 1))

    # compute fairness evaluation scores
    fairness_eval = FairnessEvaluation(test_dataset.gender, test_dataset.label, np.argmax(y_pred_test, 1))

    # print results
    print("*****************************************************************")
    print(f"Evaluation on Test Set:")
    print("Test cost: %2.3g" %cost_test)
    print("Test accuracy: %2.3g" %test_accuracy)
    print("Test UAR: %3.4g" %test_acc_uw)
    print('Test Confusion Matrix:["ang","sad","hap","neu"]')
    print(test_conf)
    print('Fairness Scores (in terms of equal opportunities):')
    fairness_eval.print_equal_opportunities()
    print("*****************************************************************")

    if return_fairness_eval:
        return fairness_eval

if __name__=='__main__':

    for i in range(30):
        model = ACRNN()
        model = train(model, f"locked_dropout_activation_reg_twin_exp{i}", num_epoch=100, verbose=True, save_model=False, return_best_model=True)
        test_dataset = IEMOCAPEval(partition='test')
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
        criterion = nn.CrossEntropyLoss()

        fairness_eval = test(model, test_loader, test_dataset, criterion, return_fairness_eval=True)
        with open(f'log/fairness_experiments/fairness_eval_twin_exp{i}.pkl', 'wb') as f:
            pickle.dump(fairness_eval, f)
        

