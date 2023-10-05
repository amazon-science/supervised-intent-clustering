# Supervised Intent Clustering
# This is a package to fine-tune language models in order to create clustering-friendly embeddings.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Optional

import omegaconf
import hydra

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from src.pl_data_modules import BasePLDataModule
from src.pl_modules import BasePLModule
from src.utils.utilities import gpus, enable_16precision, set_determinism_the_old_way


def train(conf: omegaconf.DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.train.seed)
    set_determinism_the_old_way(True)

    # data module declaration
    pl_data_module: BasePLDataModule = BasePLDataModule(conf)

    # main module declaration
    pl_module: BasePLModule = BasePLModule(conf)

    path_saved_model: str = conf.evaluation.model_checkpoint_path
    assert path_saved_model is None, "override the model_checkpoint_path in conf/"
    pl_module = pl_module.load_from_checkpoint(path_saved_model)
    hydra.utils.log("Restored saved model")

    # callbacks declaration
    callbacks_store = [RichProgressBar()]

    # trainer
    trainer: Trainer = hydra.utils.instantiate(
        conf.train.pl_trainer,
        callbacks=callbacks_store,
        gpus=gpus(conf),
        precision=enable_16precision(conf),
    )

    # module test
    trainer.test(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
