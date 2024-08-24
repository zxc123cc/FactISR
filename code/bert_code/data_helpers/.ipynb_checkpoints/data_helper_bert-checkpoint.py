import argparse
import csv
from typing import Any, Optional, Tuple

import torch
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BertTokenizer
import random
from functools import partial
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


def create_dataloaders(args,train_data,dev_data,tokenizer):
    train_dataset = ClassificationDataset(args,train_data,tokenizer)
    val_dataset = ClassificationDataset(args,dev_data,tokenizer)
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.train_batch_size,
                                        sampler=train_sampler,
                                        drop_last=False,
                                        collate_fn=train_dataset.pad_collate)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False,
                                      collate_fn=val_dataset.pad_collate)
    return train_dataloader, val_dataloader

class ClassificationDataset(Dataset):

    def __init__(self, args, data, tokenizer) -> object:
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = args.max_length

    def __len__(self) -> int:
        return len(self.data)

    def convert_list_to_str(self,evidence):
        evidence_list = [v['text'] if 'text' in v else '' for k, v in evidence.items()][:2]
        evidence_list = [f'"证据{i + 1}: {evidence}"' for i, evidence in enumerate(evidence_list)]
        now_str = '[' + ','.join(evidence_list) + ']'
        return now_str

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        if 'text' not in self.data[idx]['evidence']:
            evidence_list = self.data[idx]['evidence']
            evidence_str = self.convert_list_to_str(evidence_list)
        else:
            evidence_str = self.data[idx]['evidence']['text']
        
        input_text = sample['claim'] + ' # ' + evidence_str
        label = sample['label']
        return input_text, label

    def pad_collate(self, batch):
        data = {}
        input_text, label = zip(*batch)
        tokenizer_output = self.tokenizer.batch_encode_plus(
            input_text, padding=True, return_tensors='pt', truncation=True, max_length=self.max_length
        )
        input_ids, attention_mask = tokenizer_output.input_ids, tokenizer_output.attention_mask

        data['input_ids'] = input_ids
        data['attention_mask'] = attention_mask
        data['label'] = torch.LongTensor(label)

        return data