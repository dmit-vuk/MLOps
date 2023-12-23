from typing import List, Tuple

import mlflow
import numpy as np
import onnx
import pandas as pd
import torch
from dvc import api
from hydra import compose, initialize
from torch import nn
from torch.utils.data import DataLoader

from .data import MnistData
from .models import MLPNN


def model_mlflow(
    in_features: int,
    hidden_sizes: List[int],
    test_input: torch.Tensor,
    model_path_onnx: str,
):
    model_onnx = onnx.load_model(model_path_onnx)

    model = MLPNN(
        in_features=in_features,
        hidden_sizes=hidden_sizes,
    )
    model.eval()

    with mlflow.start_run():
        signature = mlflow.models.infer_signature(
            test_input.numpy(),
            model(test_input).detach().numpy(),
        )
        model_info = mlflow.onnx.log_model(
            model_onnx,
            "model",
            signature=signature,
        )
    return model_info


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
    return y_pred, y_true


def infer_model():
    url = 'https://github.com/dmit-vuk/MLOps'
    fs = api.DVCFileSystem(url, rev='main')
    fs.get("./MLOps/data", "./", recursive=True, download=True)
    initialize(version_base="1.3", config_path="../configs")
    config = compose("config.yaml")

    data_module = MnistData(batch_size=config.model_parameters.batch_size)
    model = MLPNN(
        in_features=config.model.in_features,
        hidden_sizes=config.model.hidden_sizes,
    )
    model.load_state_dict(torch.load(config.model.model_path))
    val_dataloader = data_module.train_dataloader()

    y_pred, y_true = evaluate(val_dataloader, model)
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df.to_csv("predictions.csv", index=False)
    print('Accuracy: ', (y_pred == y_true).sum() / len(y_true))


def infer_model_onnx():
    url = 'https://github.com/dmit-vuk/MLOps'
    fs = api.DVCFileSystem(url, rev='main')
    fs.get("./MLOps/data", "./", recursive=True, download=True)
    initialize(version_base="1.3", config_path="../configs")
    config = compose("config.yaml")

    mlflow.set_tracking_uri(config.artifacts.log_uri)
    mlflow.set_experiment(experiment_name=config.artifacts.experiment_name)

    test_input = torch.ones((1, 784))
    model_info = model_mlflow(
        config.model.in_features,
        config.model.hidden_sizes,
        test_input,
        config.model.model_path_onnx,
    )
    onnx_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)

    data_module = MnistData(batch_size=config.model_parameters.batch_size)
    val_dataloader = data_module.train_dataloader()
    y_true, y_pred = np.array([]), np.array([])
    for images, labels in val_dataloader:
        images = torch.nn.Flatten()(images)
        images = images.detach().numpy()
        predicted = onnx_pyfunc.predict(images)['probas']

        y_true = np.append(y_true, labels)
        y_pred = np.append(y_pred, predicted.argmax(axis=1))

    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df.to_csv("predictions.csv", index=False)
    print('Accuracy: ', (y_pred == y_true).sum() / len(y_true))


if __name__ == "__main__":
    infer_model()
    infer_model_onnx()
