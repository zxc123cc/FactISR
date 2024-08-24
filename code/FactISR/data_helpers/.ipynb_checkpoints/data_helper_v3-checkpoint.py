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
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.fact_verification_prompt = ("你是一个事实核查领域事实验证的专家，擅长根据证据判断新闻或某个说法的真实性。"
                                         "我将提供当前要判断的说法以及相应的证据，你要根据证据判断真实性。"
                                         "注意，你只能输出三种答案：支持/反驳/证据不足。这三种答案分别代表：证据支持当前说法，当前说法是正确的/证据反驳当前说法，当前说法是错误的/证据信息不充分，无法判读当前说法的真实性。"
                                         "再次强调，你的输出只有三种：支持、反驳以及证据不足。"
                                         "当前说法为：{}"
                                         "对应证据为：{}"
                                         "请你判断该说法的真实性：")
        self.explanation_generation_prompt = ("你是一个事实核查领域解释生成的专家，擅长根据证据和真实性生成相应的解释。"
                                              "我将提供当前说法、相应的证据以及当前说法真实性，你要根据证据和真实性，生成一段解释，说明当前说法为什么是对的/错的/证据不足或不充分的。"
                                              "注意，你的解释要严格按照证据和真实性来。"
                                              "当前说法为：{}"
                                              "对应证据为：{}"
                                              "该说法的真实性为{}"
                                              "请你生成解释：")

        self.data = self.get_data(data)

    def get_data(self, data):
        my_data = []
        label_map = {
            0: '支持',
            1: '反驳',
            2: '证据不足'
        }
        for index in range(len(data)):
            claim = data[index]['claim']
            if 'text' not in data[index]['evidence']:
                evidence_list = data[index]['evidence']
                evidence_str = self.convert_list_to_str(evidence_list)
            else:
                evidence_str = data[index]['evidence']['text']
                
            evidence_str = self.get_truncation_text(evidence_str,
                                                    tranc_len=self.max_source_length - 500)  # Reserve 100 length for instruction

            fact_verification_data = data[index].copy()
            explanation_generation_data = data[index].copy()

            fact_verification_input = self.fact_verification_prompt.format(claim, evidence_str)
            label = label_map[data[index]['label']]
            fact_verification_data['input'] = fact_verification_input
            fact_verification_data['label'] = label
            my_data.append(fact_verification_data)

            explanation_generation_input = self.explanation_generation_prompt.format(claim, evidence_str, label)
            explanation = data[index]['human_revision']
            explanation_generation_data['input'] = explanation_generation_input
            explanation_generation_data['label'] = explanation
            my_data.append(explanation_generation_data)

        return my_data

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
        context = self.data[index]['input']
        label = self.data[index]['label']

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
    def __init__(self, data, val_idx, tokenizer, max_source_length, max_target_length, task_mode):
        super(CEG4EVAL, self).__init__()
        self.val_idx = val_idx
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.task_mode = task_mode
        self.fact_verification_prompt = ("你是一个事实核查领域事实验证的专家，擅长根据证据判断新闻或某个说法的真实性。"
                                         "我将提供当前要判断的说法以及相应的证据，你要根据证据判断真实性。"
                                         "注意，你只能输出三种答案：支持/反驳/证据不足。这三种答案分别代表：证据支持当前说法，当前说法是正确的/证据反驳当前说法，当前说法是错误的/证据信息不充分，无法判读当前说法的真实性。"
                                         "再次强调，你的输出只有三种：支持、反驳以及证据不足。"
                                         "当前说法为：{}"
                                         "对应证据为：{}"
                                         "请你判断该说法的真实性：")
        self.explanation_generation_prompt = ("你是一个事实核查领域解释生成的专家，擅长根据证据和真实性生成相应的解释。"
                                              "我将提供当前说法、相应的证据以及当前说法真实性，你要根据证据和真实性，生成一段解释，说明当前说法为什么是对的/错的/证据不足或不充分的。"
                                              "注意，你的解释要严格按照证据和真实性来。"
                                              "当前说法为：{}"
                                              "对应证据为：{}"
                                              "该说法的真实性为{}"
                                              "请你生成解释：")
        self.data = self.get_data(data)

    def __len__(self):
        return len(self.data)

    def get_data(self, data):
        my_data = []
        for index in range(len(data)):
            claim = data[index]['claim']
            if 'text' not in data[index]['evidence']:
                evidence_list = data[index]['evidence']
                evidence_str = self.convert_list_to_str(evidence_list)
            else:
                evidence_str = data[index]['evidence']['text']
            evidence_str = self.get_truncation_text(evidence_str,
                                                    tranc_len=self.max_source_length - 500)  # Reserve 100 length for instruction

            if self.task_mode == 'FVP':
                fact_verification_data = data[index].copy()
                fact_verification_input = self.fact_verification_prompt.format(claim, evidence_str)
                fact_verification_data['input'] = fact_verification_input
                my_data.append(fact_verification_data)

            elif self.task_mode == 'EGP':
                explanation_generation_data = data[index].copy()
                pseudo_fact_label = data[index]['pseudo_fact_label']
                explanation_generation_input = self.explanation_generation_prompt.format(claim, evidence_str,
                                                                                         pseudo_fact_label)
                explanation_generation_data['input'] = explanation_generation_input
                my_data.append(explanation_generation_data)

        return my_data

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
        context = self.data[index]['input']
        return {
            "input_ids": context,
            'idx': idx
        }
