from functools import lru_cache

import numpy as np
import torch
from dvc import api
from hydra import compose, initialize
from tritonclient.http import (
    InferenceServerClient,
    InferInput,
    InferRequestedOutput,
)
from tritonclient.utils import np_to_triton_dtype

from .data import MnistData


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_infer(input: np.ndarray):
    triton_client = get_client()

    triton_input = InferInput(
        name="images",
        shape=input.shape,
        datatype=np_to_triton_dtype(input.dtype),
    )

    triton_input.set_data_from_numpy(input, binary_data=True)

    triton_output = InferRequestedOutput("probas", binary_data=True)
    query_response = triton_client.infer(
        "model", [triton_input], outputs=[triton_output]
    )

    output = query_response.as_numpy("probas")

    return output


def test():
    url = 'https://github.com/dmit-vuk/MLOps'
    fs = api.DVCFileSystem(url, rev='main')
    fs.get("./MLOps/data", "./", recursive=True, download=True)
    initialize(version_base="1.3", config_path="../configs")
    config = compose("config.yaml")
    data_module = MnistData(batch_size=config.model_parameters.batch_size)
    val_dataloader = data_module.train_dataloader()
    for images, labels in val_dataloader:
        images = torch.nn.Flatten()(images)
        images = images.detach().numpy()

        y_pred = call_triton_infer(images).argmax(axis=1)
        y_true = labels
        break
    print('Batch accuracy: ', (y_pred == y_true.numpy()).sum() / len(y_true))


if __name__ == "__main__":
    test()
