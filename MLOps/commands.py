import fire

from mlops import infer as infer_module
from mlops import train as train_module


def train():
    train_module.train_model()


def infer():
    infer_module.infer_model()


def run_server():
    infer_module.infer_model_onnx()


if __name__ == '__main__':
    fire.Fire()
