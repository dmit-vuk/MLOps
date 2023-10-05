import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor

from typing import Tuple, List

from models import MLPNN

def get_dataloader(batch_size: int, train: bool) -> Tuple[DataLoader]:
    dataset = torchvision.datasets.MNIST(root="data", train=train, download=True, transform=ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataloader


def train_epoch(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer, device: str) -> None:
    model.train()
    for idx, data in enumerate(dataloader):
        optimizer.zero_grad()

        images, labels = data
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    
def evaluate(dataloader: DataLoader, model: nn.Module, loss_fn, device: str) -> Tuple[float]:
    model.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    y_pred, y_true = torch.Tensor(), torch.Tensor()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            total_loss += float(loss_fn(outputs, labels).detach())
            _, predicted = torch.max(outputs.data, 1)
            total_accuracy += predicted.eq(labels.data).cpu().sum()

            y_pred = torch.cat([y_pred, predicted.cpu()])
            y_true = torch.cat([y_true, labels.cpu()])
    
    return total_loss / len(dataloader.dataset), total_accuracy / len(dataloader.dataset), y_pred, y_true
    

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
    best_test_accuracy = 0
    for epoch in range(num_epochs):
        train_epoch(train_loader, model, loss_fn, optimizer, device)
        
        train_loss, train_acc, _, _ = evaluate(train_loader, model, loss_fn, device)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        
        test_loss, test_acc, _, _ = evaluate(test_loader, model, loss_fn, device)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)

        if best_test_accuracy < test_acc:
            best_test_accuracy = test_acc
            torch.save(model.state_dict(), 'best_params.pth')
        
        print(
            'Epoch: {0:d}/{1:d}. Loss (Train/Test): {2:.3f}/{3:.3f}. Accuracy (Train/Test): {4:.3f}/{5:.3f}'.format(
                epoch + 1, num_epochs, train_losses[-1], test_losses[-1], train_accuracies[-1], test_accuracies[-1]
            )
        )
    return train_losses, train_accuracies, test_losses, test_accuracies

if __name__ == '__main__':
    train_loader = get_dataloader(batch_size=512, train=True)
    test_loader = get_dataloader(batch_size=512, train=False)

    model = MLPNN(in_features=784, hidden_sizes = [1024, 512, 128, 10])
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    train(train_loader=train_loader, test_loader=test_loader, model=model, 
          loss_fn=loss, optimizer=optimizer, device='cpu', num_epochs=10)