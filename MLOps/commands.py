import fire

from mlops import infer, train


def train_file():
    train.train_model()


def infer_file():
    infer.infer_model()


if __name__ == '__main__':
    fire.Fire()
