# Supervised Intent Clustering
# This is a package to fine-tune language models in order to create clustering-friendly embeddings.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    'base_language_models/paraphrase-multilingual-mpnet-base-v2',
    cache_folder="")