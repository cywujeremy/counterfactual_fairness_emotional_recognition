import time
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

from data.emotion_labels import EMOTIONAL_LABELS, EMOTIONAL_LABELS_SHORT
from model.acrnn import ACRNN
from dataset.datasets import IEMOCAPSamples, IEMOCAPSmall


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Device: ", device)

BATCH_SIZE = 10
LEARNING_RATE = 1e-5
EPOCHS = 300
PATH = "data/data_tiny"
NUM_CLASSES = 4


def train(model, train_loader, optimizer, criterion, scaler, epoch):
    
    model.train()
    
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False)
    total_loss = 0
    num_correct = 0
    
    for batch_idx, (X, y) in enumerate(train_loader):
        
        X, y = X.permute((0, 3, 1, 2)).float().cuda(), y.long().cuda()
        
        optimizer.zero_grad()
        
        with autocast():
            
            output = model(X)
            loss = criterion(output, y)

        num_correct += int((torch.argmax(output, axis=1) == y).sum())
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
    
    batch_bar.close()
    print(f"Epoch {epoch}/{EPOCHS}: Train Loss {float(total_loss / len(train_loader)):.04f}, Learning Rate {float(optimizer.param_groups[0]['lr']):.06f}")

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
    val_data = IEMOCAPSmall(partition="dev")
    test_data = IEMOCAPSmall(partition="test")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model = ACRNN(num_classes=num_classes, p=batch_size)
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    
    for epoch in range(1, epochs + 1):
        
        train(model, train_loader, optimizer, criterion, scaler, epoch)

if __name__ == "__main__":
    
    main()

