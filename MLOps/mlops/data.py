import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class MnistData(pl.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()

        self.batch_size = batch_size
        self.train_dataset = torchvision.datasets.MNIST(
            root="data", train=True, download=False, transform=ToTensor()
        )
        self.test_dataset = torchvision.datasets.MNIST(
            root="data", train=False, download=False, transform=ToTensor()
        )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=True
        )
        return dataloader
