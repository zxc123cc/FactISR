import logging
import sys
import time
import json
from torch.cuda.amp import autocast

sys.path.append('models')
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import tqdm
from transformers import BertTokenizer
import transformers

transformers.logging.set_verbosity_error()
from utils import prepare_optimizer, prepare_scheduler, prepare_optimizer_delamination
from models.model_classification import BertForSequenceClassification
from data_helpers.data_helper_bert import create_dataloaders
from evaluate_classification_bert import evaluate
from config import parse_args
from utils import init_distributed_mode, setup_logging, setup_device, setup_seed


def train_and_validate(args):
    device = torch.device(args.device)
    print('device: ', device)
    print("create dataloader")
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_dir)

    with open(args.train_file, 'r') as f:
        train_data = json.load(f)
    with open(args.dev_file, 'r') as f:
        dev_data = json.load(f)
    train_dataloader, val_dataloader = create_dataloaders(args, train_data, dev_data, tokenizer)
    print("load model")

    model = BertForSequenceClassification.from_pretrained(args.pretrain_model_dir)
    model.to(device)

    optimizer = prepare_optimizer(model, args.learning_rate, args.weight_decay, args.eps)

    scheduler = prepare_scheduler(optimizer, args.max_epochs, len(train_dataloader), args.warmup_ratio,
                                  gradient_accumulation_steps=args.gradient_accumulation_steps)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    best_score = 0
    step, global_step = 0, 0
    print("start training")
    logging.info(f"start time >>> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    for epoch in range(args.max_epochs):
        model.train()
        with tqdm.tqdm(total=train_dataloader.__len__(), desc=f"[{epoch + 1}] / [{args.max_epochs}]training... ") as t:
            for index, batch in enumerate(train_dataloader):
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                label_ids = batch['label'].to(args.device)

                output_dict = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=label_ids
                                    )
                loss = output_dict.get("loss")
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                step += 1
                t.set_postfix(loss=loss.cpu().item())
                t.update(1)

        meters = evaluate(args, model, val_dataloader)
        score = meters['macro_f1']
        logging.info(
            f"acc = {meters['acc']},  macro_f1 = {meters['macro_f1']}, macro_recall = {meters['macro_recall']}, macro_precision = {meters['macro_precision']}")

        if score >= best_score:
            best_score = score
            model_path = os.path.join(args.save_model_path, f'model_best.bin')
            state_dict = model.state_dict()
            torch.save(state_dict, model_path)

        torch.cuda.empty_cache()

def test(args):
    device = torch.device(args.device)
    print('device: ', device)
    print("create dataloader")
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_dir)

    with open(args.train_file, 'r') as f:
        train_data = json.load(f)
    with open(args.test_file, 'r') as f:
        test_data = json.load(f)
    train_dataloader, val_dataloader = create_dataloaders(args, train_data, test_data, tokenizer)
    print("load model")

    model = BertForSequenceClassification.from_pretrained(args.pretrain_model_dir)
    model.load_state_dict(torch.load(os.path.join(args.save_model_path, f'model_best.bin'), map_location='cpu'))
    model.to(device)
    
    print("start inference")
    logging.info(f"test time >>> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    meters = evaluate(args, model, val_dataloader)
    logging.info(
        f"acc = {meters['acc']},  macro_f1 = {meters['macro_f1']}, macro_recall = {meters['macro_recall']}, macro_precision = {meters['macro_precision']}")


if __name__ == '__main__':
    args = parse_args()
    # 设置显卡
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    # os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    if args.distributed:
        init_distributed_mode(args)

    os.makedirs(args.save_model_path, exist_ok=True)

    setup_logging(args)
    setup_device(args)
    setup_seed(args)

    logging.info("Training/evaluation parameters: %s", args)
    train_and_validate(args)
    test(args)
