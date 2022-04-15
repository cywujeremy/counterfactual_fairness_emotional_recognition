import time
import pickle
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummaryX import summary
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score

from data.emotion_labels import EMOTIONAL_LABELS, EMOTIONAL_LABELS_SHORT
from model.acrnn import ACRNN, ACRNN2
from dataset.datasets import IEMOCAPSamples, IEMOCAPSmall


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Device: ", device)

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 300
PATH = "data/data"
NUM_CLASSES = 11


def train(model, train_loader, optimizer, criterion, scaler, scheduler, epoch, train_accuracy=None, train_uar=None):
    
    model.train()
    
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False)
    total_loss = 0
    num_correct = 0
    y_pred = torch.tensor([]).cuda()
    y_true = torch.tensor([]).cuda()
    
    for batch_idx, (X, y) in enumerate(train_loader):
        
        X, y = X.float().cuda(), y.long().cuda()
        
        X = X.permute((1, 0, 3, 2))

        optimizer.zero_grad()
        
        with autocast():
            output = model(X)
            loss = criterion(output, y)
        
        y_pred_batch = output.argmax(axis=1)
        y_pred = torch.concat((y_pred, y_pred_batch))
        y_true = torch.concat((y_true, y))

        num_correct += int((y_pred_batch == y).sum())
        total_loss += float(loss)
        batch_bar.set_postfix(
            accuracy=f"{float(num_correct / ((batch_idx + 1) * BATCH_SIZE)) * 100:.4f}%",
            loss=f"{float(total_loss / (batch_idx + 1)):.4f}",
            lr=f"{float(optimizer.param_groups[0]['lr']):.6f}"
        )

        batch_bar.update()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    scheduler.step()
    train_accuracy.append(accuracy_score(y_true.cpu(), y_pred.cpu()))
    train_uar.append(recall_score(y_true.cpu(), y_pred.cpu(), average='macro'))

    batch_bar.close()
    print(f"Epoch {epoch}/{EPOCHS}: Train Loss={float(total_loss / len(train_loader)):.04f}, Learning Rate={float(optimizer.param_groups[0]['lr']):.06f}, Accuracy={float(num_correct / ((batch_idx + 1) * BATCH_SIZE)) * 100:.4f}%")

def validate(model, val_loader, epoch, val_accuracy=None, val_uar=None):

    model.eval()

    num_correct = 0
    num_samples = 0
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, leave=False)
    y_pred = torch.tensor([]).cuda()
    y_true = torch.tensor([]).cuda()

    for batch_idx, (X, y) in enumerate(val_loader):

        X, y = X.float().cuda(), y.long().cuda()

        X = X.permute((1, 0, 3, 2))

        with torch.no_grad():
            output = model(X)

        y_pred_batch = output.argmax(axis=1)
        y_pred = torch.concat((y_pred, y_pred_batch))
        y_true = torch.concat((y_true, y))

        num_samples += len(output)
        num_correct += int(torch.sum(y == y_pred_batch))
        running_accuracy = num_correct / num_samples

        batch_bar.set_postfix(
            accuracy=f"{float(running_accuracy) * 100:.4f}%"
        )

        batch_bar.update()
    
    val_accuracy.append(running_accuracy)
    unweighted_average_recall = recall_score(y_true.cpu(), y_pred.cpu(), average='macro')
    val_uar.append(unweighted_average_recall)
    
    batch_bar.close()
    print(f"Validation after Epoch {epoch}/{EPOCHS}: validation accuracy={float(running_accuracy) * 100:.4f}%, UAR={unweighted_average_recall:.4f}")

def test(model, test_loader):

    with open("data/IEMOCAP_val.pkl", "rb") as f:
        data = pickle.load(f)
    gender_labels = data[0][428:]

    model.eval()

    num_correct = 0
    num_samples = 0
    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, leave=False)
    y_pred = torch.tensor([]).cuda()
    y_true = torch.tensor([]).cuda()

    for batch_idx, (X, y) in enumerate(test_loader):

        X, y = X.permute((0, 3, 1, 2)).float().cuda(), y.long().cuda()

        with torch.no_grad():
            output = model(X)

        y_pred_batch = output.argmax(axis=1)
        y_pred = torch.concat((y_pred, y_pred_batch))
        y_true = torch.concat((y_true, y))

        num_samples += len(output)
        num_correct += int(torch.sum(y == y_pred_batch))
        running_accuracy = num_correct / num_samples

        batch_bar.set_postfix(
            accuracy=f"{float(running_accuracy) * 100:.4f}%"
        )

        batch_bar.update()
    
    unweighted_average_recall = recall_score(y_true.cpu(), y_pred.cpu(), average='macro')

    y_true = np.array(y_true.cpu())
    y_pred = np.array(y_pred.cpu())

    pred_df = pd.DataFrame({'gender': gender_labels,
                            'y_true': y_true,
                            'y_pred': y_pred})
    
    batch_bar.close()
    print(f"Test Result: Test accuracy={float(running_accuracy) * 100:.4f}%, Test UAR={unweighted_average_recall:.4f}")

    output_name = f"pred_df_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    pred_df.to_csv(f'results/{output_name}.csv')

def main():
    
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    epochs = EPOCHS
    path = PATH
    num_classes = NUM_CLASSES
    
    # train_data = IEMOCAPSamples(path, partition='train')
    # val_data = IEMOCAPSamples(path, partition='dev', shuffle=False)
    # test_data = IEMOCAPSamples(path, partition='test', shuffle=False)

    train_data = IEMOCAPSmall(partition='train')
    val_data = IEMOCAPSmall(pickle_path="data/IEMOCAP_val.pkl", partition="dev")
    test_data = IEMOCAPSmall(pickle_path="data/IEMOCAP_val.pkl", partition="test")
    
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data.collate_fn,
    #                           num_workers=8, pin_memory=True)
    # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=val_data.collate_fn,
    #                         num_workers=8, pin_memory=True)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,collate_fn=test_data.collate_fn,
    #                          num_workers=8, pin_memory=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             num_workers=16, pin_memory=True)
    
    model = ACRNN2()
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scaler = GradScaler()
    
    train_accuracy = []
    train_uar = []
    validation_accuracy = []
    validation_uar = []

    # train_loader = list(train_loader)

    for epoch in range(1, epochs + 1):
        
        train(model, train_loader, optimizer, criterion, scaler, scheduler, epoch, train_accuracy, train_uar)
        validate(model, val_loader, epoch, validation_accuracy, validation_uar)

    # test(model, test_loader)

    # training_log_df = pd.DataFrame({'epoch': list(range(EPOCHS)),
    #                                 'train_accuracy': train_accuracy,
    #                                 'val_accuracy': validation_accuracy,
    #                                 'train_uar': train_uar,
    #                                 'val_uar': validation_uar})
    
    # training_log_name = f"log_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    # training_log_df.to_csv(f"log/{training_log_name}.csv")

if __name__ == "__main__":
    
    main()

