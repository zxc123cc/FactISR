import json
from sklearn import metrics
import numpy as np
from torch.utils.data import Dataset
import random
import numpy as np
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.utils import PaddingStrategy
from imblearn.under_sampling import RandomUnderSampler


class CEGDataSet(Dataset):
    '''
        iteratively return the profile of each author
    '''

    def __init__(self, data, tokenizer, max_source_length, max_target_length):
        super(CEGDataSet, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        random.shuffle(self.data)
        self.instruct = "你是一个事实核查专家，擅长判断新闻或某个说法的真实性，并且生成相应的解释。{}请根据证据判断真实性并给出解释。"

    def __len__(self):
        return len(self.data)
    
    def convert_list_to_str(self, evidence):
        evidence_list = [v['text'] if 'text' in v else '' for k, v in evidence.items()][:2]
        evidence_list = [f'"证据{i + 1}: {evidence}"' for i, evidence in enumerate(evidence_list)]
        now_str = '[' + ','.join(evidence_list) + ']'
        return now_str
    
    def get_truncation_text(self, text, tranc_len=200):
        text_ids = self.tokenizer(
            [text], add_special_tokens=False,
            return_tensors='pt')['input_ids'][0][:tranc_len]
        text = self.tokenizer.decode(text_ids)
        return text
    
    def __getitem__(self, index):
        claim = self.data[index]['claim']
        if 'text' not in self.data[index]['evidence']:
            evidence_list = self.data[index]['evidence']
            evidence_str = self.convert_list_to_str(evidence_list)
        else:
            evidence_str = self.data[index]['evidence']['text']
        label = self.data[index]['human_revision']
        
        now_input = f'当前说法为：{claim}\n当前证据为：{evidence_str}\n'
        now_input = self.get_truncation_text(now_input,tranc_len=self.max_source_length-100) # Reserve 100 length for instruction
        
        context = self.instruct.format(now_input)

        input_ids = self.tokenizer.encode(text=context, add_special_tokens=True, truncation=True,
                                          max_length=self.max_source_length)
        label_ids = self.tokenizer.encode(text=label, add_special_tokens=False, truncation=True,
                                          max_length=self.max_target_length)
        
        labels = [-100] * len(input_ids) + label_ids + [self.tokenizer.eos_token_id]
        input_ids = input_ids + label_ids + [self.tokenizer.eos_token_id]

        return {
            "input_ids": input_ids,
            "labels": labels
        }


@dataclass
class DataCollatorForCEG:
    """
        borrow and modified from transformers.DataCollatorForSeq2Seq
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        # breakpoint()
        features = self.tokenizer.pad(
            features,
            padding=True,
            max_length=max_label_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        # breakpoint() # [(len(features[i]['input_ids']),len(features[i]['labels'])) for i in range(4)]
        return features


class CEG4EVAL(Dataset):
    def __init__(self, datas, val_idx, tokenizer,max_source_length, max_target_length, model_type):
        super(CEG4EVAL, self).__init__()
        self.datas = datas
        self.val_idx = val_idx
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type
        self.instruct = "你是一个事实核查专家，擅长判断新闻或某个说法的真实性，并且生成相应的解释。{}请根据证据判断真实性并给出解释。"
        
    def __len__(self):
        return len(self.datas)

    def convert_list_to_str(self, evidence):
        evidence_list = [v['text'] if 'text' in v else '' for k, v in evidence.items()][:2]
        evidence_list = [f'"证据{i + 1}: {evidence}"' for i, evidence in enumerate(evidence_list)]
        now_str = '[' + ','.join(evidence_list) + ']'
        return now_str
    
    def get_truncation_text(self, text, tranc_len=200):
        text_ids = self.tokenizer(
            [text], add_special_tokens=False,
            return_tensors='pt')['input_ids'][0][:tranc_len]
        text = self.tokenizer.decode(text_ids)
        return text

    def __getitem__(self, index):
        idx = self.val_idx[index]
        claim = self.datas[index]['claim']
        if 'text' not in self.datas[idx]['evidence']:
            evidence_list = self.datas[idx]['evidence']
            evidence_str = self.convert_list_to_str(evidence_list)
        else:
            evidence_str = self.datas[idx]['evidence']['text']
        
        now_input = f'当前说法为：{claim}\n当前证据为：{evidence_str}\n'
        now_input = self.get_truncation_text(now_input,tranc_len=self.max_source_length-100) # Reserve 100 length for instruction
        
        context = self.instruct.format(now_input)
        
        return {
            "input_ids": context,
            'idx': idx
        }
