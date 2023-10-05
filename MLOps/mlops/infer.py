import pandas as pd
import torch
from torch import nn

from models import MLPNN
from train import get_dataloader,  evaluate

if __name__ == '__main__':
    test_loader = get_dataloader(batch_size=512, train=False)

    model = MLPNN(in_features=784, hidden_sizes = [1024, 512, 128, 10])
    model.load_state_dict(torch.load('best_params.pth'))
    loss = nn.CrossEntropyLoss()
    test_loss, test_acc , y_pred, y_true = evaluate(dataloader=test_loader, model=model, loss_fn=loss, device='cpu')
    print('Test loss: ', test_loss)
    print('Test accuracy: ', test_acc.item())
    
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    df = pd.DataFrame({'true': y_true, 'pred': y_pred}) #convert to a dataframe
    df.to_csv("predictions.csv", index=False)