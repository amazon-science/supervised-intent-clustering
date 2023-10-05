# Supervised Intent Clustering
# This is a package to fine-tune language models in order to create clustering-friendly embeddings.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import *
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import nn
from torch.nn import functional as F

class SupervisedClusteringLoss(torch.nn.Module):
    def __init__(self, C: float, r: float) -> None:
        super().__init__()
        self.C = C
        self.r = r

    def forward(self,
        pairwise_score_matrics: torch.Tensor, 
        intents: torch.Tensor, device: str):

        pairwise_class_equality = torch.eq(
            intents[None, :], intents[:, None]).float()
            
        pairwise_class_equality_negative = torch.ne(
            intents[None, :], intents[:, None]).float()

        gold_similarity_matrix = -pairwise_score_matrics*pairwise_class_equality
        viol_similarity_matrix = (
            pairwise_score_matrics
            + pairwise_class_equality_negative*self.C*self.r 
            - pairwise_class_equality*self.C)
        
        viol_similarity_matrix = -viol_similarity_matrix*(viol_similarity_matrix>0).float()

        viol_spanning_tree = minimum_spanning_tree(viol_similarity_matrix.cpu().detach().numpy()).toarray()
        gold_spanning_tree = minimum_spanning_tree(gold_similarity_matrix.cpu().detach().numpy()).toarray()
        
        viol_spanning_tree = torch.Tensor(viol_spanning_tree).to(device)
        gold_spanning_tree = torch.Tensor(gold_spanning_tree).to(device)

        a = torch.count_nonzero(gold_spanning_tree)
        b = torch.count_nonzero(viol_spanning_tree*pairwise_class_equality)
        c = torch.count_nonzero(viol_spanning_tree*pairwise_class_equality_negative)
        
        delta = a -b +c*self.r

        viol_spanning_tree[viol_spanning_tree!=0] = 1
        viol_score = torch.sum(pairwise_score_matrics*viol_spanning_tree)
        gold_spanning_tree[gold_spanning_tree!=0] = 1
        gold_score = torch.sum(pairwise_score_matrics*gold_spanning_tree) 

        obj = self.C*delta + viol_score - gold_score

        if not delta > 0:
            loss = torch.Tensor([0]).to(device)
        else:
            loss = torch.max(torch.Tensor([0]).to(device), obj)
        
        return loss

class BalancedBinaryCrossEntropy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self,
        distance_matrix: torch.Tensor, 
        intents: torch.Tensor):
    
            pairwise_class_equality = torch.eq(
                intents[None, :], intents[:, None]).float()

            pos_samples = torch.sum(pairwise_class_equality)-len(intents)
            neg_samples = torch.sum(1 - pairwise_class_equality)
            distance_after_sigmoid = self.sigmoid(distance_matrix)

            positive_loss = (torch.sum(
                pairwise_class_equality*torch.log(distance_after_sigmoid))-len(intents))/pos_samples
            negative_loss = torch.sum(
                (1-pairwise_class_equality)*(torch.log(1-distance_after_sigmoid)))/neg_samples
            
            loss = -(positive_loss+negative_loss)

            return loss


class CosineSimLoss(torch.nn.Module):
    # loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.
    def __init__(self) -> None:
        super().__init__()

    def forward(self,
        distance_matrix: torch.Tensor, 
        intents: torch.Tensor):

        pairwise_class_equality = torch.eq(
            intents[None, :], intents[:, None]).float()

        loss_individual_scores = (pairwise_class_equality - distance_matrix).pow(2)

        pos_samples = torch.sum(pairwise_class_equality)-len(intents)
        neg_samples = torch.sum(1 - pairwise_class_equality)

        positive_loss = (torch.sum(
            pairwise_class_equality*loss_individual_scores)-len(intents))/pos_samples
        negative_loss = torch.sum(
            (1-pairwise_class_equality)*loss_individual_scores)/neg_samples
            
        loss = (positive_loss+negative_loss)

        return loss

class ContrastLoss(torch.nn.Module):
    # losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
    def __init__(self, margin: float) -> None:
        super().__init__()
        self.margin = margin

    def forward(self,
        distance_matrix: torch.Tensor, 
        intents: torch.Tensor):


        pairwise_class_equality = torch.eq(
            intents[None, :], intents[:, None]).float()

        pos_samples = torch.sum(pairwise_class_equality)-len(intents)
        neg_samples = torch.sum(1 - pairwise_class_equality)

        positive_loss = torch.sum(
            pairwise_class_equality*distance_matrix.pow(2))/pos_samples

        negative_loss = torch.sum(
            (1-pairwise_class_equality)*F.relu(
                self.margin-distance_matrix).pow(2))/neg_samples

        loss = (positive_loss+negative_loss)
            
        return loss

