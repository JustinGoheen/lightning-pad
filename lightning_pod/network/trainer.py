"""
This module serves to consolidate basic use cases of PyTorch Lightning and to test
functionality of this template using an encoder-decoder on MNIST. Documentation links are provided below.

example code: https://pytorch-lightning.readthedocs.io/en/latest/model/train_model_basic.html
Trainer: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
Reproducibility: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#reproducibility
TensorBoardLogger: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loggers.tensorboard.html#tensorboard
SimpleProfiler: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.profiler.SimpleProfiler.html#simpleprofiler
ModelCheckpoint: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.ModelCheckpoint.html#modelcheckpoint
"""

import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_pod.network.module import LitModel
from lightning_pod.pipeline.dataloader import get_data


if __name__ == "__main__":

    # GET CURRENT WORKING DIRECTORY
    cwd = os.getcwd()
    # SET LOGGER
    logs_dir = os.path.join(cwd, "logs")
    logger = TensorBoardLogger(logs_dir, name="lightning_logs")
    # SET PROFILER
    profile_dir = os.path.join(logs_dir, "profiler")
    profiler = SimpleProfiler(dirpath=profile_dir, filename="profiler", extended=True)
    # SET CHECKPOINT DIRECTORY
    chkpt_dir = os.path.join(cwd, "models", "checkpoints")
    checkpoint_callback = ModelCheckpoint(dirpath=chkpt_dir, filename="model")
    # SET CALLBACKS
    callbacks = [checkpoint_callback]
    # SET SEED
    seed_everything(42, workers=True)
    #  GET DATALOADER
    train_loader, test_loader = get_data(return_loader=True, split=True, num_workers=0)
    #  SET MODEL
    model = LitModel()
    # SET TRAINER
    trainer = Trainer(
        max_epochs=5,
        limit_train_batches=0.10,  # use only x% of training samples
        enable_checkpointing=True,  # default
        deterministic=True,  # for reproducibility
        # strategy="ddp",  # set a strategy when training in Grid
        accelerator="auto",
        devices="auto",
        logger=logger,
        profiler=profiler,
        callbacks=callbacks,
    )
    # TRAIN MODEL https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#fit
    trainer.fit(model=model, train_dataloaders=train_loader)
    # PERSIST MODEL
    pretrained_dir = os.path.join(cwd, "models", "production")
    modelpath = os.path.join(pretrained_dir, "model.onnx")
    input_sample = train_loader.dataset[0][0]
    model.to_onnx(modelpath, input_sample=input_sample, export_params=True)
