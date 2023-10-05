# Supervised Intent Clustering
# This is a package to fine-tune language models in order to create clustering-friendly embeddings.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
from typing import *

import hydra
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping # TODO: re-add RichProgressBar
from pytorch_lightning.loggers import WandbLogger


def set_fast_dev_run(conf: DictConfig):
    if conf.train.pl_trainer.fast_dev_run:
        print(f"Debug mode <{conf.train.pl_trainer.fast_dev_run}>. Forcing debugger configuration")
        # Debuggers don't like GPUs nor multiprocessing
        conf.train.pl_trainer.gpus = 0
        conf.train.pl_trainer.precision = 32
        conf.data.num_workers = 0
        # Switch wandb mode to offline to prevent online logging
        conf.logging.wandb_arg.mode = "offline"


def gpus(conf: DictConfig) -> int:
    """Utility to determine the number of GPUs to use."""
    return conf.train.pl_trainer.gpus if torch.cuda.is_available() else 0


def enable_16precision(conf: DictConfig) -> int:
    """Utility to determine the number of GPUs to use."""
    return conf.train.pl_trainer.precision if torch.cuda.is_available() else 32


def set_determinism_the_old_way(deterministic: bool) -> None:
    # determinism for cudnn
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            # fixing non-deterministic part of horovod
            # https://github.com/PyTorchLightning/pytorch-lightning/pull/1572/files#r420279383
            os.environ["HOROVOD_FUSION_THRESHOLD"] = str(0)


def get_number_of_cpu_cores() -> int:
    return os.cpu_count()


def build_callbacks(conf: DictConfig) -> Tuple[List[pl.Callback], Optional[ModelCheckpoint]]:
    """
    Add here your pytorch lightning callbacks
    """
    callbacks_store = [] # TODO: re-add RichProgressBar()
    if conf.train.early_stopping_callback is not None:
        early_stopping_callback: EarlyStopping = hydra.utils.instantiate(conf.train.early_stopping_callback)
        callbacks_store.append(early_stopping_callback)

    model_checkpoint_callback: Optional[ModelCheckpoint] = None
    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(conf.train.model_checkpoint_callback)
        callbacks_store.append(model_checkpoint_callback)
    return callbacks_store, model_checkpoint_callback


def build_wandb_logger(conf):
    logger: Optional[WandbLogger] = None
    if conf.logging.log and not conf.train.pl_trainer.fast_dev_run:
        hydra.utils.log(f"Instantiating Wandb Logger")
        logger = hydra.utils.instantiate(conf.logging.wandb_arg)
    return logger