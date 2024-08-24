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


def convert_list_to_str(evidence):
    evidence_list = [v['text'] if 'text' in v else '' for k, v in evidence.items()][:2]
    evidence_list = [f'"证据{i + 1}: {evidence}"' for i, evidence in enumerate(evidence_list)]
    now_str = '[' + ','.join(evidence_list) + ']'
    return now_str

# with open('../final_data/CHEF/train.json','r') as f:
#     train_datas = json.load(f)

# example_input_list = ['说法：' + data['claim'] + '\n' +'证据：' + convert_list_to_str(data['evidence']) for data in train_datas]

with open('../../data/TrendFact/train.json','r') as f:
    train_datas = json.load(f)

example_input_list = ['说法：' + data['claim'] + '\n' +'证据：' + data['evidence']['text'] for data in train_datas]

example_output_list = [data['human_revision'] for data in train_datas]


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
                         "接下来，我将提供几个例子，你可以参照例子进行输出\n"
                         f"示例1输入：{example_input_list[0]}\n"
                         f"示例1输出：{example_output_list[0]}\n"
                         f"示例2输入：{example_input_list[1]}\n"
                         f"示例2输出：{example_output_list[1]}\n"
                         f"示例3输入：{example_input_list[2]}\n"
                         f"示例3输出：{example_output_list[2]}\n"
                         f"示例4输入：{example_input_list[3]}\n"
                         f"示例4输出：{example_output_list[3]}\n"
                         f"示例5输入：{example_input_list[4]}\n"
                         f"示例5输出：{example_output_list[4]}\n"
                         "当前说法为：{}\n"
                         "当前证据为：{}\n"
                         "###\n"
                         "请你根据当前说法和给定证据生成一个解释。\n"
                         "注意：不要使用第一人称!\n"
                         "注意：内容精简，不包含冗余信息!\n"
                         "注意：严格根据证据进行真实性判断，并在解释的最后进行输出，例如“因此，该说法是错误的”\n"
                         "注意：严格基于证据进行解释生成!\n")
        
        # self.system_prompt = ("你是一个事实核查专家，擅长判断新闻或某个说法的真实性，并且生成相应的解释。\n"
        #                  "已知事实核查任务是根据给定的证据来判断当前说法的真实性以及生成相应的解释，包含三种类型（正确/错误/证据不足）。\n"
        #                  "接下来，我将提供几个例子，你可以参照例子进行输出\n"
        #                  f"示例1输出：{example_output1}\n"
        #                  f"示例2输出：{example_output2}\n"
        #                  f"示例3输出：{example_output3}\n"
        #                  f"示例4输出：{example_output4}\n"
        #                  f"示例5输出：{example_output5}\n"   
        #                  "请你根据当前说法和给定证据生成一个解释。\n" 
        #                  "注意：不要使用第一人称!\n"
        #                  "注意：内容精简，不包含冗余信息!\n"
        #                  "注意：严格根据证据进行真实性判断，并在解释的最后进行输出，例如“因此，该说法是错误的”\n"
        #                  "注意：严格基于证据进行解释生成!\n")
        self.system_prompt = ("你是一个事实核查专家，擅长判断新闻或某个说法的真实性，并且生成相应的解释。\n"
                         "已知事实核查任务是根据给定的证据来判断当前说法的真实性以及生成相应的解释，包含三种类型（正确/错误/证据不足）。\n"
                         "接下来，我将提供几个例子，你可以参照例子进行输出\n"
                         f"示例1输入：{example_input_list[0]}\n"
                         f"示例1输出：{example_output_list[0]}\n"
                         f"示例2输入：{example_input_list[1]}\n"
                         f"示例2输出：{example_output_list[1]}\n"
                         f"示例3输入：{example_input_list[2]}\n"
                         f"示例3输出：{example_output_list[2]}\n"
                         f"示例4输入：{example_input_list[3]}\n"
                         f"示例4输出：{example_output_list[3]}\n"
                         f"示例5输入：{example_input_list[4]}\n"
                         f"示例5输出：{example_output_list[4]}\n"
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
        
        if self.model_type == 'glm3' or self.model_type == 'baichuan2':
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
