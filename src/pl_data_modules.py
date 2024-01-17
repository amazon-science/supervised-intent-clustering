# Supervised Intent Clustering
# This is a package to fine-tune language models in order to create clustering-friendly embeddings.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Any, Union, List, Optional

from omegaconf import DictConfig

import hydra
import torch
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data.sampler import BatchSampler

from intent_classification_dataset import *

class BasePLDataModule(pl.LightningDataModule):
    """
    FROM LIGHTNING DOCUMENTATION

    A DataModule standardizes the training, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)

    A DataModule implements 5 key methods:

    * **prepare_data** (things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode).
    * **setup**  (things to do on every accelerator in distributed mode).
    * **train_dataloader** the training dataloader.
    * **val_dataloader** the val dataloader(s).
    * **test_dataloader** the test dataloader(s).


    This allows you to share a full dataset without explaining how to download,
    split transform and process the data

    """

    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = conf

    def prepare_data(self, *args, **kwargs):
        print(f"self.conf.data.train_path: {self.conf.data.train_path}")
        self.train_dataset = IntentDataset(
            hydra.utils.to_absolute_path(self.conf.data.train_path), 
            self.conf.model.sentence_transformer_model_name,
            self.conf.data.language_list)
        print(f"self.conf.data.validation_path: {self.conf.data.validation_path}")
        self.dev_dataset = IntentDataset(
            hydra.utils.to_absolute_path(self.conf.data.validation_path), 
            self.conf.model.sentence_transformer_model_name,
            self.conf.data.language_list)
        print(f"self.conf.data.test_path")
        self.test_dataset = IntentDataset(
            hydra.utils.to_absolute_path(self.conf.data.test_path), 
            self.conf.model.sentence_transformer_model_name,
            self.conf.data.language_list)

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        print(f"train_dataset: {self.train_dataset}")
        balanced_batch_sampler = BalancedBatchSampler(
            self.train_dataset, self.conf.data.intent_classes_per_batch, 
            self.conf.data.samples_per_class_per_batch)
        train_dataloader = DataLoader(
            self.train_dataset, 
            batch_sampler=balanced_batch_sampler,
            num_workers=self.conf.data.num_workers)
        return train_dataloader

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        balanced_batch_sampler = BalancedBatchSampler(
            self.dev_dataset, self.conf.data.intent_classes_per_batch, 
            self.conf.data.samples_per_class_per_batch)
        val_dataloader = DataLoader(
            self.dev_dataset, 
            batch_sampler=balanced_batch_sampler,
            num_workers=self.conf.data.num_workers)
        return val_dataloader

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        balanced_batch_sampler = BalancedBatchSampler(
            self.test_dataset, self.conf.data.intent_classes_per_batch, 
            self.conf.data.samples_per_class_per_batch)
        test_dataloader = DataLoader(
            self.test_dataset, 
            batch_sampler=balanced_batch_sampler,
            num_workers=self.conf.data.num_workers)
        return test_dataloader

class BalancedBatchSampler(BatchSampler):  #TODO: discuss sampling strategy more in depth
    """
    BatchSampler samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """
    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for item in loader:
            self.labels_list.append(item['intent'])
        self.labels = torch.LongTensor(self.labels_list)  #HELP : modify
        self.labels_set = list(set(self.labels.numpy()))
        print(len(self.labels_set))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // (self.batch_size)

@hydra.main(config_path="../conf", config_name="root")
def main(conf: DictConfig):
    dataset = IntentDataset(
        hydra.utils.to_absolute_path(conf.data.train_path), # validation_path / train_path / test_path
        conf.model.sentence_transformer_model_name,
        conf.data.language_list)
    
    balanced_batch_sampler = BalancedBatchSampler(dataset, 8, 5)
    
    dataloader = DataLoader(dataset, batch_sampler=balanced_batch_sampler)

    for key, value in balanced_batch_sampler.label_to_indices.items():
        print(key, len(value))

    my_testiter = iter(dataloader)
    
    counter = 0
    for batch in my_testiter:
        print(batch['intent'])
        counter += 1
        print(counter)
    
    print(len(balanced_batch_sampler))

if __name__ == "__main__":
    main()