import git
import pytorch_lightning as pl
import torch
from dvc import api
from hydra import compose, initialize

from .data import MnistData
from .models import MLPNN


class TrainerModule(pl.LightningModule):
    def __init__(self, conf, git_commit_id: str):
        super().__init__()
        self.save_hyperparameters()

        self.model = MLPNN(
            in_features=conf.model.in_features,
            hidden_sizes=conf.model.hidden_sizes,
        )

        self.loss = torch.nn.CrossEntropyLoss()
        self.learning_rate = conf.training.learning_rate

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.loss(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        accuracy = predicted.eq(labels.data).cpu().sum() / len(predicted)

        metrics = {'loss': loss.detach(), 'accuracy': accuracy}
        self.log_dict(
            metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.loss(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        accuracy = predicted.eq(labels.data).cpu().sum() / len(predicted)

        metrics = {'loss': loss.detach(), 'accuracy': accuracy}
        self.log_dict(
            metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        """Define optimizers and LR schedulers."""
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)


def load_data(url: str = './..'):
    fs = api.DVCFileSystem(url, rev='main')
    print(fs.find('/', detail=False, dvc_only=True))
    fs.get("../data", "./data", recursive=True)


def train_model():
    load_data()
    initialize(version_base="1.3", config_path="../configs")
    config = compose("config.yaml")
    repo = git.Repo(search_parent_directories=True)

    data_module = MnistData(batch_size=config.training.batch_size)
    train_module = TrainerModule(config, git_commit_id=repo.head.object.hexsha)

    logger = pl.loggers.MLFlowLogger(
        experiment_name=config.artifacts.experiment_name,
        tracking_uri=config.artifacts.log_uri,
        # save_dir = "./logs/mlruns"
    )
    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        max_epochs=config.training.epochs,
        logger=logger,
    )

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.train_dataloader()
    trainer.fit(train_module, train_dataloader, val_dataloader)

    # loss_history, acc_history = trainer.train(batch_size=64, epch_num=32)
    train_module.save_model('model.pth')


if __name__ == "__main__":
    train_model()
