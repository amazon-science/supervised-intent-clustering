# Supervised Intent Clustering
# This is a package to fine-tune language models in order to create clustering-friendly embeddings.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import *

import omegaconf
import hydra
import json
from os import path

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from src.pl_data_modules import BasePLDataModule
from src.pl_modules import BasePLModule
from src.utils.utilities import set_determinism_the_old_way, set_fast_dev_run, build_callbacks, build_wandb_logger, gpus


def train(conf: omegaconf.DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.train.seed)
    set_determinism_the_old_way(True)

    # if fast dev run enable in the config we disable the logger, num workers and gpus
    set_fast_dev_run(conf)

    # data module declaration
    pl_data_module: BasePLDataModule = BasePLDataModule(conf)
    pl_data_module.setup()
    pl_data_module.prepare_data()

    # main module declaration
    pl_module: BasePLModule = BasePLModule(conf)

    # callbacks declaration
    callbacks_store, model_checkpoint_callback = build_callbacks(conf)

    logger: Optional[WandbLogger] = build_wandb_logger(conf)

    # trainer
    trainer: Trainer = hydra.utils.instantiate(
        conf.train.pl_trainer, callbacks=callbacks_store, logger=logger, gpus=gpus(conf))

    experiment_results = {
        'pre_training_metrics': {},
        'post_training_metrics': {}
    }
    
    # model validate - pre-training
    
    train_dataloader = pl_data_module.train_dataloader()
    train_output = trainer.validate(pl_module, dataloaders=train_dataloader)
    
    experiment_results[
        'pre_training_metrics'][
            'train_set'] = {
                your_key[4:]: train_output[0][your_key] for your_key in ['val_AUPRC', 'val_loss']}


    val_dataloader = pl_data_module.val_dataloader()
    val_output = trainer.validate(pl_module, dataloaders=val_dataloader)
    
    experiment_results[
        'pre_training_metrics'][
            'dev_set'] = {
                your_key[4:]: val_output[0][your_key] for your_key in ['val_AUPRC', 'val_loss']}
    
    test_dataloader = pl_data_module.test_dataloader()
    test_output = trainer.validate(pl_module, dataloaders=test_dataloader)
    
    experiment_results[
        'pre_training_metrics'][
            'test_set'] = {
                your_key[4:]: test_output[0][your_key] for your_key in ['val_AUPRC', 'val_loss']}
    
    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)

    if conf.train.model_checkpoint_callback is not None and not trainer.fast_dev_run:
        pl_module = pl_module.load_from_checkpoint(checkpoint_path=model_checkpoint_callback.best_model_path)

    # module test
    #trainer.test(pl_module, datamodule=pl_data_module)
    
    # model validate - post-training
    train_dataloader = pl_data_module.train_dataloader()
    train_output = trainer.validate(pl_module, dataloaders=train_dataloader)
    
    experiment_results[
        'post_training_metrics'][
            'train_set'] = {
                your_key[4:]: train_output[0][your_key] for your_key in ['val_AUPRC', 'val_loss']}

    val_dataloader = pl_data_module.val_dataloader()
    val_output = trainer.validate(pl_module, dataloaders=val_dataloader)
    
    experiment_results[
        'post_training_metrics'][
            'dev_set'] = {
                your_key[4:]: val_output[0][your_key] for your_key in ['val_AUPRC', 'val_loss']}
    
    test_dataloader = pl_data_module.test_dataloader()
    test_output = trainer.validate(pl_module, dataloaders=test_dataloader)
    
    experiment_results[
        'post_training_metrics'][
            'test_set'] = {
                your_key[4:]: test_output[0][your_key] for your_key in ['val_AUPRC', 'val_loss']}

    # train end
    if conf.logging.log:
        logger.experiment.finish()

    if conf.model.training_objective == 'contrastive_learning':
        loss_hyperparameter = conf.model.contrastive_margin
    elif conf.model.training_objective == 'triplet_margin_loss':
        loss_hyperparameter = conf.model.triplet_margin
    elif conf.model.training_objective == 'supervised_learning':
        loss_hyperparameter = f'{conf.model.C}_{conf.model.r}'
    else:
        loss_hyperparameter = 'xxx'
        
    pl_module.base_multilingual_sent_encoder.save_pretrained(
        hydra.utils.to_absolute_path(
            f'fine_tuned_language_models/{conf.data.experiment_round}/{conf.model.base_model_name}_{conf.data.dataset}_{conf.data.modality}_{conf.model.training_objective}_{loss_hyperparameter}')
    )
    pl_data_module.train_dataset.tokenizer.save_pretrained(
        hydra.utils.to_absolute_path(
            f'fine_tuned_language_models/{conf.data.experiment_round}/{conf.model.base_model_name}_{conf.data.dataset}_{conf.data.modality}_{conf.model.training_objective}_{loss_hyperparameter}')
    )
    
    to_dump = {
        f'{conf.model.base_model_name}_{conf.data.dataset}_{conf.data.modality}_{conf.model.training_objective}_{loss_hyperparameter}_{conf.data.intent_classes_per_batch}_{conf.data.samples_per_class_per_batch}': experiment_results
    }
    
    experiment_file = hydra.utils.to_absolute_path("experiment_metrics.txt")
    
    with open(experiment_file, 'a') as f:
            f.write(json.dumps(to_dump, indent=3) + "\n")
            
#     path_results = "/".join(model_checkpoint_callback.best_model_path.split('/')[:-2])
#     with open(hydra.utils.to_absolute_path(f"{path_results}/experiment_metrics.json"), "w") as writer:
#         json.dump(experiment_results, writer, indent=4, sort_keys=True)

@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()

