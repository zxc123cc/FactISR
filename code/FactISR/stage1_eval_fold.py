# -*- coding: utf-8 -*-

import os
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    BitsAndBytesConfig,
    AutoModelForCausalLM
)
from peft import PeftModel,get_peft_model
import torch
from data_helpers.data_helper_v3 import CEG4EVAL
import json
from accelerate import Accelerator
from tqdm import tqdm
from metrics import evaluate_metrics
import argparse
from sklearn.model_selection import KFold
import gc

_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',default='ZhipuAI/chatglm3-6b-32k')
parser.add_argument('--lora_path',default='ZhipuAI/chatglm3-6b-32k')
parser.add_argument('--test_path', help='The path to the pub file',default='test_pub.json')
parser.add_argument('--saved_dir',default='../test_result')
parser.add_argument('--save_name',default='test_result.json')
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--max_source_length',type=int,default=30000)
parser.add_argument('--max_target_length',type=int,default=16)
parser.add_argument('--model_type',default='glm')
parser.add_argument('--task_mode',default='FVP')
parser.add_argument('--fold',type=int,default=0)
parser.add_argument('--fact_label_path',default='test_result.json')
parser.add_argument('--num_beams',type=int,default=1)

args = parser.parse_args()

set_seed(args.seed)

accelerator = Accelerator()
device = torch.device(0)

batch_size = 1

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

result = []

if args.task_mode == 'EGP':
    with open(args.fact_label_path,'r') as f:
        pseudo_fact_label_dict = json.load(f)

    def add_pseudo_fact_label(data, idx):
        data['pseudo_fact_label'] = pseudo_fact_label_dict[str(idx)]
        return data

with open(args.test_path,'r') as f:
    datas = json.load(f)

now_fold, k_fold = 0, 5
kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)


for train_index, val_index in kf.split(datas):
    if now_fold != args.fold:
        now_fold += 1
        continue
    if args.task_mode == 'FVP':
        val_fold = [datas[i] for i in val_index]
    else:
        val_fold = [add_pseudo_fact_label(datas[i], int(i)) for i in val_index]
    val_index = [int(i) for i in val_index]
    eval_dataset = CEG4EVAL(
        val_fold, val_index,
        tokenizer = tokenizer,
        max_source_length = args.max_source_length,
        max_target_length = args.max_target_length,
        task_mode = args.task_mode
    )
    now_fold += 1

print('done reading dataset')

model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_8bit=False, trust_remote_code=True).half()
lora_model = PeftModel.from_pretrained(model, args.lora_path).half()
print('done loading peft model')

def collate_fn(batch):
    batch = {k: [item[k] for item in batch] for k in ('input_ids','idx')}
    batch_input = tokenizer(
        batch['input_ids'],
        padding='longest',
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return batch_input, batch['idx']

dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size = batch_size ,collate_fn=collate_fn)
val_data = accelerator.prepare_data_loader(dataloader, device_placement=True)
model = accelerator.prepare_model(model)

print('len val data: ', len(val_data))

with torch.no_grad():
    for index,batch in enumerate(tqdm(val_data)):
        batch_input, idx = batch
        if args.model_type == 'glm3' or args.model_type == 'glm4':
            generated_ids = model.generate(**batch_input, max_length=batch_input['input_ids'].shape[-1] + args.max_target_length,num_beams=args.num_beams)
        elif args.model_type == 'Yi1_5':
            generated_ids = model.generate(
                batch_input.input_ids,
                attention_mask=batch_input.attention_mask,
                max_new_tokens=args.max_target_length,
                eos_token_id=tokenizer.eos_token_id
            )
        else:
            generated_ids = model.generate(
                batch_input.input_ids,
                max_new_tokens=args.max_target_length
            )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(batch_input.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        node_result = [[idx[i],response[i]] for i in range(batch_size)]
        batch_result = accelerator.gather_for_metrics(node_result)
        if accelerator.is_main_process:
            result.extend(batch_result)

del model, lora_model, dataloader, val_data, val_fold
torch.cuda.empty_cache()
gc.collect()

if accelerator.is_main_process:
    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)
    res_list = {}
    for i in result:
        [idx,response] = i
        if idx not in res_list.keys():
            res_list[idx] = {}
        res_list[idx] = response
    save_path = os.path.join(args.saved_dir,args.save_name)
    with open(save_path, 'w') as f:
        json.dump(res_list, f)
