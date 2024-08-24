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


class INDDataSet(Dataset):
    '''
        iteratively return the profile of each author
    '''

    def __init__(self, dataset, tokenizer, max_source_length, max_target_length):
        super(INDDataSet, self).__init__()
        self.texts, self.labels = dataset
        self.data = [{'text': text, 'label': label} for text, label in zip(self.texts, self.labels)]
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        random.shuffle(self.data)
        self.instruct = "{}, \n###\nGive me an answer between 'yes' or 'no'."

        self.yes_token = self.tokenizer.encode(text='yes', add_special_tokens=False, truncation=True, )
        self.no_token = self.tokenizer.encode(text='no', add_special_tokens=False, truncation=True, )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        now_data = self.data[index]
        text = now_data['text']
        label = now_data['label']
        context = self.instruct.format(text)

        input_ids = self.tokenizer.encode(text=context, add_special_tokens=True, truncation=True,
                                          max_length=self.max_source_length)
        label_ids = self.yes_token if int(label) else self.no_token
        input_ids = input_ids + label_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * (len(input_ids) - 2) + label_ids + [self.tokenizer.eos_token_id]

        return {
            "input_ids": input_ids,
            "labels": labels
        }


@dataclass
class DataCollatorForIND:
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
        self.instruct = ("你是一个事实核查专家，擅长判断新闻或某个说法的真实性，并且生成相应的解释。\n"
                         "已知事实核查任务是根据给定的证据来判断当前说法的真实性以及生成相应的解释，包含三种类型（正确/错误/证据不足）。\n"
                         "现在我提供给你当前说法以及证据，请你给出相应的解释。\n"
                         "当前说法为：{}\n"
                         "当前证据为：{}\n"
                         "###\n"
                         "请你根据当前说法和给定证据生成一个解释。\n"
                         "注意：不要使用第一人称!\n"
                         "注意：内容精简，不包含冗余信息!\n"
                         "注意：严格根据证据进行真实性判断，并在解释的最后进行输出，例如“因此，该说法是错误的”\n"
                         "注意：严格基于证据进行解释生成!\n")
        
        self.system_prompt = ("你是一个事实核查专家，擅长判断新闻或某个说法的真实性，并且生成相应的解释。\n"
                         "已知事实核查任务是根据给定的证据来判断当前说法的真实性以及生成相应的解释，包含三种类型（正确/错误/证据不足）。\n"
                         "请你根据当前说法和给定证据生成一个解释。\n"
                         "注意：不要使用第一人称!\n"
                         "注意：内容精简，不包含冗余信息!\n"
                         "注意：严格根据证据进行真实性判断，并在解释的最后进行输出，例如“因此，该说法是错误的”\n"
                         "注意：严格基于证据进行解释生成!\n")
        
    def __len__(self):
        return len(self.datas)

    def convert_list_to_str(self, evidence):
        evidence_list = [v['text'] for k, v in evidence.items()][:2]
        evidence_list = [f'"证据{i + 1}: {evidence}"' for i, evidence in enumerate(evidence_list)]
        now_str = '[' + ','.join(evidence_list) + ']'
        return now_str

    def __getitem__(self, index):
        idx = self.val_idx[index]
        claim = self.datas[index]['claim']
        if 'text' not in self.datas[idx]['evidence']:
            evidence_list = self.datas[idx]['evidence']
            evidence_str = self.convert_list_to_str(evidence_list)
        else:
            evidence_str = self.datas[idx]['evidence']['text']
        
        # context = self.instruct.format(claim, evidence_str)
        
        if self.model_type == 'glm3' or self.model_type == 'baichuan2' or self.model_type == 'deepseek_v2':
            context = self.instruct.format(claim, evidence_str)
        else:
            user_input = f"当前说法为：{claim}\n当前证据为：{evidence_str}\n"
            messages = [
                {"role": "system", "content": f"{self.system_prompt}"},
                {"role": "user", "content": user_input}
            ]
            context = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        return {
            "input_ids": context,
            'idx': idx
        }
