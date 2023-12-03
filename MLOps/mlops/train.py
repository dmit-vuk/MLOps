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
        self.learning_rate = conf.model_parameters.learning_rate

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
            self.model.parameters(), lr=self.learning_rate
        )
        return optimizer

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)


def train_model():
    url = 'https://github.com/dmit-vuk/MLOps'
    fs = api.DVCFileSystem(url, rev='main')
    fs.get("./MLOps/data", "./", recursive=True, download=True)
    initialize(version_base="1.3", config_path="../configs")
    config = compose("config.yaml")
    try:
        repo = git.Repo(search_parent_directories=True)
        git_commit_id = repo.head.object.hexsha
    except git.InvalidGitRepositoryError:
        git_commit_id = "not a git repo"

    data_module = MnistData(batch_size=config.model_parameters.batch_size)
    train_module = TrainerModule(config, git_commit_id=git_commit_id)

    logger = pl.loggers.MLFlowLogger(
        experiment_name=config.artifacts.experiment_name,
        tracking_uri=config.artifacts.log_uri,
    )

    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        max_epochs=config.model_parameters.epochs,
        logger=logger,
    )

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.train_dataloader()
    trainer.fit(train_module, train_dataloader, val_dataloader)

    # loss_history, acc_history = trainer.train(batch_size=64, epch_num=32)
    train_module.save_model(config.model.model_path)

    train_module.model.eval()
    test_input = torch.ones((1, 784))
    torch.onnx.export(
        train_module.model,
        test_input,
        config.model.model_path_onnx,
        export_params=True,
        # opset_version=15,
        # do_constant_folding=True,
        input_names=["images"],
        output_names=["probas"],
        dynamic_axes={
            "images": {0: "BATCH_SIZE"},
            "probas": {0: "BATCH_SIZE"},
        },
    )


if __name__ == "__main__":
    train_model()
