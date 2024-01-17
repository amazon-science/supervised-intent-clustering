# Supervised Intent Clustering
# This is a package to fine-tune language models in order to create clustering-friendly embeddings.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import *
import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
import hydra

class IntentDataset(Dataset):
    def __init__(self,dataset_path: str, model_name: str, list_of_languages: List[str]):
        self.tokenizer = AutoTokenizer.from_pretrained(
            hydra.utils.to_absolute_path(model_name))
        self.pre_filtered_intent_dataset = pd.read_csv(
            dataset_path, usecols = ['utterance_intent', 'utterance_text', 'utterance_lang'])
        self.intent_dataset = self.pre_filtered_intent_dataset[
            self.pre_filtered_intent_dataset['utterance_lang'].isin(list_of_languages)]
        self.utterance_intents = self.intent_dataset['utterance_intent'].tolist()
        self.encoded_utterances = self.tokenize_utterances(
            self.intent_dataset['utterance_text'].tolist(), model_name)
        self.intent_encoding = self.encode_intents(
            self.intent_dataset['utterance_intent'].tolist()
        )


    def __getitem__(self, index) -> Dict[str, Union[str, int]]:
        intent = self.utterance_intents[index]
        encoded_intent = self.intent_encoding[intent]

        #print("input_ids:", self.encoded_utterances["input_ids"])
        #print("index:", index, "type(index):", type(index))

        item = {
            'utterance_encoding': {
                'input_ids': self.encoded_utterances['input_ids'][index],
                'attention_mask': self.encoded_utterances['attention_mask'][index],
                },
            'intent': encoded_intent,
        }

        return item

    def __len__(self):
        return len(self.intent_dataset)

    def tokenize_utterances(
        self, list_of_utterannces: List,
        model_name: str):
        
        encoded_utterances = self.tokenizer(
            list_of_utterannces,padding=True, truncation=True, return_tensors='pt')
        return encoded_utterances
        #for row_num, utterance in enumerate(list_of_utterannces, start=1):
        #    print(f"Tokenizing {row_num}...")
        #    self.tokenizer([utterance], padding=True, truncation=True, return_tensors='pt')
        #return []
    
    def encode_intents(self, utterance_intent_list: List):
        for row_num, intent in enumerate(utterance_intent_list, start=1):
            if type(intent) == float:
                print(f"{row_num} {intent}")
        intents = list(set(utterance_intent_list))

        #print("intents:", set([type(intent) for intent in intents]))
        intents.sort()
        intent_encoding = {}
        counter = 0          #  TODO: refactor with collections.Counter
        for intent in intents:
            intent_encoding[intent] = counter
            counter += 1
        return intent_encoding


if __name__ == '__main__':
    dataset = IntentDataset(
        'data/dev.csv', 
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    print(len(dataset))
    print(dataset[0])

    loader = DataLoader(dataset, batch_size=2)

    my_testiter = iter(loader)
    
    what = my_testiter.next()

    print(what)