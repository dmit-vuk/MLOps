from typing import Tuple

import pandas as pd
import torch
from hydra import compose, initialize
from torch import nn
from torch.utils.data import DataLoader

from .data import MnistData
from .models import MLPNN


def evaluate(dataloader: DataLoader, model: nn.Module) -> Tuple[float]:
    model.eval()
    y_pred, y_true = torch.Tensor(), torch.Tensor()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            images, labels = data
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            y_pred = torch.cat([y_pred, predicted.cpu()])
            y_true = torch.cat([y_true, labels.cpu()])

    return (
        y_pred,
        y_true,
    )


def infer():
    initialize(version_base="1.3", config_path="../configs")
    config = compose("config.yaml")

    data_module = MnistData(batch_size=config.training.batch_size)
    model = MLPNN(
        in_features=config.model.in_features,
        hidden_sizes=config.model.hidden_sizes,
    )
    model.load_state_dict('model.pth')
    val_dataloader = data_module.train_dataloader()

    y_pred, y_true = evaluate(val_dataloader, model)
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    infer()
