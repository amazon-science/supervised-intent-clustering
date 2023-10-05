# Supervised Intent Clustering
# This is a package to fine-tune language models in order to create clustering-friendly embeddings.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import re
import pandas as pd

results_list = []

with open('experiment_metrics.txt', 'r', encoding='utf-8') as file:
    json_data = re.sub(r"}\s*{", "},{", file.read())
    results_list.extend(json.loads("[" + json_data + "]"))
    
results_post_processing = []

for model_metrics in results_list:
    model_name = list(model_metrics.keys())[0]
    experiment_dict = {'model_name': model_name}
    for data_split in ["train", "dev", "test"]:
        experiment_dict[f"{data_split}_pre_training_AUPRC"] = model_metrics[model_name][
            "pre_training_metrics"][f"{data_split}_set"]["AUPRC"]
        experiment_dict[f"{data_split}_pre_training_loss"] = model_metrics[model_name][
            "pre_training_metrics"][f"{data_split}_set"]["loss"]
        experiment_dict[f"{data_split}_post_training_AUPRC"] = model_metrics[model_name][
            "post_training_metrics"][f"{data_split}_set"]["AUPRC"]
        experiment_dict[f"{data_split}_post_training_loss"] = model_metrics[model_name][
            "post_training_metrics"][f"{data_split}_set"]["loss"]
    results_post_processing.append(experiment_dict)

print(results_post_processing)

df = pd.DataFrame(results_post_processing)

print(df)

df.to_csv(
        'experiment_metrics_post_analysis.csv', decimal=',', float_format='%.3f')