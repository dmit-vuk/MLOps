import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import Tuple, List

from models import models
#from data

def train_epoch(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer, device: str) -> None:
    model.train()
    for idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        
        tokens, tokens_lens, ratings = data['tokens'], data['tokens_lens'], data['ratings']
        tokens, tokens_lens, ratings = tokens.to(device), tokens_lens.to(device), ratings.to(device)
            
        outputs = model(tokens, tokens_lens)
        loss = loss_fn(outputs, ratings)
        loss.backward()
        optimizer.step()
    
def evaluate(dataloader: DataLoader, model: nn.Module, loss_fn, device: str) -> Tuple[float]:
    model.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            tokens, tokens_lens, ratings = data['tokens'], data['tokens_lens'], data['ratings']
            tokens, tokens_lens, ratings = tokens.to(device), tokens_lens.to(device), ratings.to(device)
            outputs = model(tokens, tokens_lens)
            
            total_loss += float(loss_fn(outputs, ratings).detach())
            _, predicted = torch.max(outputs.data, 1)
            total_accuracy += predicted.eq(ratings.data).cpu().sum()
    
    return total_loss / len(dataloader.dataset), total_accuracy / len(dataloader.dataset)
    

def train(
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        model: nn.Module, 
        loss_fn, 
        optimizer, 
        device: str, 
        num_epochs: int,
) -> Tuple[List[float]]:
    
    test_losses = []
    train_losses = []
    test_accuracies = []
    train_accuracies = []
    for epoch in range(num_epochs):
        train_epoch(train_loader, model, loss_fn, optimizer, device)
        
        train_loss, train_acc = evaluate(train_loader, model, loss_fn, device)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        
        test_loss, test_acc = evaluate(test_loader, model, loss_fn, device)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        
        print(
            'Epoch: {0:d}/{1:d}. Loss (Train/Test): {2:.3f}/{3:.3f}. Accuracy (Train/Test): {4:.3f}/{5:.3f}'.format(
                epoch + 1, num_epochs, train_losses[-1], test_losses[-1], train_accuracies[-1], test_accuracies[-1]
            )
        )
    return train_losses, train_accuracies, test_losses, test_accuracies