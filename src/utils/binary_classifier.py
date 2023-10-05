# Supervised Intent Clustering
# This is a package to fine-tune language models in order to create clustering-friendly embeddings.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List
from torch import nn
import torch

class CoreNet(nn.Module):
    def __init__(
        self, dropout: float, 
        input_dim: int):

        super().__init__()
        layers: List[nn.Module] = []

        input_dim: int = input_dim

        layers.append(nn.Linear(input_dim, 1))
#         layers.append(nn.BatchNorm1d(1)),
#         layers.append(nn.ReLU())
#         layers.append(nn.Dropout(dropout))

        self.layers: nn.Module = nn.Sequential(*layers)

    def forward(
        self, utterance_embeddings: torch.Tensor, device: str) -> torch.Tensor:

        size = torch.arange(0, len(utterance_embeddings), 
                            dtype=torch.long, device=device)
        all_pairs = torch.cartesian_prod(size, size).to(device)
        
        #print(all_pairs.device, device)
        
        idx_sx = all_pairs[:, 0]
        idx_dx = all_pairs[:, 1]

        features = torch.cat(
            (utterance_embeddings[idx_sx], 
            utterance_embeddings[idx_dx],
            torch.abs(utterance_embeddings[idx_sx]-utterance_embeddings[idx_dx])), dim=1)
        
        scores = self.layers(features).squeeze(1)

        distance_matrix = torch.sparse.FloatTensor(all_pairs.T, scores)

        return distance_matrix