import fire

from mlops import infer as infer_module
from mlops import train as train_module
from mlops import triton_client


def train():
    train_module.train_model()


def infer():
    infer_module.infer_model()


def run_server():
    infer_module.infer_model_onnx()


def run_triton():
    triton_client.test()


if __name__ == '__main__':
    fire.Fire()
