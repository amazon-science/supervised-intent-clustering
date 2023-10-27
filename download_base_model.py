# Supervised Intent Clustering
# This is a package to fine-tune language models in order to create clustering-friendly embeddings.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

# This script is meant to download sentence-transformer models to be fine-tuned.

from sentence_transformers import SentenceTransformer
import sys

model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
output_folder = './base_language_models'

if len(sys.argv) > 1:
    model_name = sys.argv[1]
    
model = SentenceTransformer(model_name)
model.save(f'{output_folder}/{model_name}')