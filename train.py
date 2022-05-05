"""
This training pipeline is adapted and reconstructed from the codes of the original paper.
Reference link: https://github.com/Chien-Hung/Speech-Emotion-Recognition
"""

from __future__ import absolute_import
from __future__ import division

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
from datasets import IEMOCAPTrain, IEMOCAPTrainAUG, IEMOCAPEval
from utils.training_tracker import TrainingTracker
from utils.fairness_eval import FairnessEvaluation

num_epoch = 300
num_classes = 4
batch_size = 128
learning_rate = 0.001

start_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
checkpoint = f'./checkpoint/{start_time}'
#experiment_name = "acrnn_locked_dropout_act_reg0.3"

#experiment_name = "fairness_dataaug_50_50"

clip = 0
ar_alpha = 0.3
device = 'cuda'

aug = True
ratio = 0.5

def train(aug, ratio):

    best_valid_uw = 0

    tracker = TrainingTracker(experiment_name)

    if aug:
        train_dataset = IEMOCAPTrainAUG(ratio=ratio)
    else:
        train_dataset = IEMOCAPTrain()
    valid_dataset = IEMOCAPEval(partition='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print('DATASET:', len(train_loader))
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

                if not os.path.isdir(checkpoint):
                    os.mkdir(checkpoint)

                model_name = f"model_{experiment_name}_{best_valid_uw:.4f}.pth"
                torch.save(model, os.path.join(checkpoint, model_name))

            # print results
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
    ratio_lst = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for r in ratio_lst:
        for i in range(5):
            experiment_name = "fairness_dataaug_ratio{}_trial{}".format(r, i)

            train(aug, ratio)
