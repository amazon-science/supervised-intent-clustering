# Supervised Intent Clustering
# This is a package to fine-tune language models in order to create clustering-friendly embeddings.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from transformers import AutoModel
from torchmetrics import F1Score, AUROC, AveragePrecision
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import LpDistance
from intent_classification_dataset import *
from pytorch_metric_learning.reducers import MeanReducer, AvgNonZeroReducer
from pytorch_metric_learning.distances import CosineSimilarity
from utils.losses import *
from utils.binary_classifier import CoreNet
import random

class BasePLModule(pl.LightningModule):
    def __init__(self, conf, *args, **kwargs) -> None:
        super().__init__()
        self.conf = conf
        self.save_hyperparameters(conf)
        self.base_multilingual_sent_encoder = AutoModel.from_pretrained(
            hydra.utils.to_absolute_path(self.conf.model.sentence_transformer_model_name))
        self.triplet_margin_loss = losses.TripletMarginLoss(
            margin=self.conf.model.triplet_margin,
            distance=LpDistance(),
            reducer=AvgNonZeroReducer()  #MeanReducer() AvgNonZeroReducer()
        )
        self.cosine_similarity = CosineSimilarity()
        self.cosine_similarity_loss = CosineSimLoss()
        self.contrastive_loss = ContrastLoss(
            margin=self.conf.model.contrastive_margin)
        self.classification_head = CoreNet(
            self.conf.model.dropout,
            self.conf.model.embeddings_length*3
        )
        self.supervised_loss = SupervisedClusteringLoss(
            C=self.conf.model.C, r=self.conf.model.r)
        
        self.balanced_BCE = BalancedBinaryCrossEntropy()
        self.f1_score = F1Score(threshold=0.8)
        self.AUROC = AUROC()
        self.AUPRC = AveragePrecision()

    def forward(self, sample) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        sample_tokenization = sample['utterance_encoding']
        sample_intents = sample['intent']
        
        model_output = self.base_multilingual_sent_encoder(**sample_tokenization)
        utterance_embeddings = self.mean_pooling(
            model_output, sample_tokenization['attention_mask'])

        if self.conf.model.training_objective == 'contrastive_learning':
            distance_matrix = 2*(1 - self.cosine_similarity(
                utterance_embeddings, utterance_embeddings))
            loss = self.contrastive_loss(
                distance_matrix, sample_intents)
        
        elif self.conf.model.training_objective == 'cosine_similarity_loss':
            distance_matrix = self.cosine_similarity(
                utterance_embeddings, utterance_embeddings)
            loss = self.cosine_similarity_loss(
                distance_matrix, sample_intents)

        elif self.conf.model.training_objective == 'triplet_margin_loss':
            loss = self.triplet_margin_loss(
                utterance_embeddings, sample_intents)
        
        elif self.conf.model.training_objective == 'binary_classification':

            distance_matrix = self.classification_head(
                utterance_embeddings, device=self.device).to_dense()

            loss = self.balanced_BCE(distance_matrix, sample_intents)

        elif self.conf.model.training_objective == 'supervised_learning':
 
            if self.conf.model.distance_matrix_for_supervised == 'cosine_distance':
                distance_matrix = self.cosine_similarity(utterance_embeddings, utterance_embeddings)
                distance_matrix = distance_matrix*100
            elif self.conf.model.distance_matrix_for_supervised == 'sigmoid_score':
                distance_matrix = self.classification_head(
                    utterance_embeddings, device=self.device).to_dense()

            loss = self.supervised_loss(
                distance_matrix, sample_intents, self.device)
       
        elif self.conf.model.training_objective == 'triplet_plus_supervised':
            
            distance_matrix = self.cosine_similarity(utterance_embeddings, utterance_embeddings)
            distance_matrix = distance_matrix*100
            
            supervised_loss = self.supervised_loss(
                distance_matrix, sample_intents, self.device)

            triplet_loss = self.triplet_margin_loss(
                utterance_embeddings, sample_intents)

            #print(100*triplet_loss)
            #print(supervised_loss/100)
            

            loss = triplet_loss + supervised_loss/1000

        else:
            raise Exception("Sorry, this is not a valid training objective")
        
        
        scores = self.cosine_similarity(
            utterance_embeddings, utterance_embeddings)
        labels = torch.eq(sample_intents[None, :], sample_intents[:, None])
        
        f1 = self.f1_score(scores, labels)
        
        AUROC = self.AUROC(
            torch.flatten(scores), 
            torch.flatten(labels))
        
        AUPRC = self.AUPRC(
            torch.flatten(scores), 
            torch.flatten(labels))
        
        output_dict = {
            'utterance_embeddings': utterance_embeddings,
            'loss': loss,
            'f1': f1,
            'AUROC': AUROC,
            'AUPRC': AUPRC
            
        }
        return output_dict

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(batch)
        self.log_dict({
            "loss": forward_output["loss"],
            "f1": forward_output["f1"],
            "AUROC": forward_output["AUROC"],
            "AUPRC": forward_output["AUPRC"]
        }, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": forward_output["loss"]}

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        forward_output = self.forward(batch)
        self.log_dict({
            "val_loss": forward_output["loss"],
            "val_f1": forward_output["f1"],
            "val_AUROC": forward_output["AUROC"],
            "val_AUPRC": forward_output["AUPRC"]
        }, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": forward_output["loss"]}
    
    # def validation_epoch_end(self, outputs):
    #     batch_losses = [x["val_loss"] for x in outputs] # This part
    #     epoch_loss = torch.stack(batch_losses).mean() 
    #     self.log("val_loss_avg", epoch_loss.item(), prog_bar=True)
    #     return {'val_loss_avg': epoch_loss.item()}

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        forward_output = self.forward(batch)
        self.log_dict({
            "test_loss": forward_output["loss"],
            "test_f1": forward_output["f1"],
            "test_AUROC": forward_output["AUROC"],
            "test_AUPRC": forward_output["AUPRC"]
        }, on_step=False, on_epoch=True, prog_bar=True)
        return {"test_loss": forward_output["loss"]}

    def configure_optimizers(self):
        """
        FROM PYTORCH LIGHTNING DOCUMENTATION

        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.

            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        only_language_model: bool = (
            self.conf.model.training_objective == 'cosine_similarity_loss'
            or self.conf.model.training_objective == 'triplet_margin_loss'
            or self.conf.model.training_objective == 'contrastive_learning' 
            or (self.conf.model.training_objective == 'supervised_learning'
                and self.conf.model.distance_matrix_for_supervised == 'cosine_distance'))
                  
        if only_language_model:
            optimizer = hydra.utils.instantiate(
                self.conf.model.optimizer, params=self.base_multilingual_sent_encoder.parameters())
        else:
            optimizer = torch.optim.AdamW(
                params=[{'params': self.classification_head.parameters(),
                        'lr': self.conf.model.optimizer.lr},
                        {'params': self.base_multilingual_sent_encoder.parameters(),
                         'lr': self.conf.model.language_model_learning_rate,
                         'weight_decay': self.conf.model.language_model_weight_decay, 'correct_bias': False}])
        
        # optimizer = hydra.utils.instantiate(
        #     self.conf.model.optimizer, params=self.base_multilingual_sent_encoder.parameters())
        return optimizer

