import json
import sys
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from fuzzywuzzy import fuzz,process


def parser(text):

    subtext_list = text.split('。')
    last_text = subtext_list[-1]
    if last_text == '':
        last_text = subtext_list[-2]
    if '正确' in last_text:
        return 0
    if '错误' in last_text:
        return 1
    return 2


def parserv2(text):
    # 定义三句话
    sentences = [
        "因此，该说法是正确的。",
        "因此，该说法是错误的。",
        "因此，证据不足以验证该说法的真实性。",
    ]
    sim_scores = []
    for sentence in sentences:
        similarity = fuzz.partial_ratio(sentence, text)
        sim_scores.append(similarity)
    # print(sim_scores)
    max_score = max(sim_scores)
    max_index = sim_scores.index(max_score)
    return max_index

def parser_FVP(text):

    subtext_list = text.split('。')
    last_text = subtext_list[-1]
    if last_text == '':
        last_text = subtext_list[-2]
    if '支持' in last_text:
        return 0
    if '反驳' in last_text:
        return 1
    return 2


def compute_metrics_fn(preds, labels):
    assert len(preds) == len(labels)
    f1 = f1_score(y_true= labels, y_pred=preds, average="macro", labels=np.unique(labels))
    acc = accuracy_score(y_true= labels, y_pred=preds)
    p = precision_score(y_true= labels, y_pred=preds, average="macro", labels=np.unique(labels))
    r = recall_score(y_true= labels, y_pred=preds, average="macro", labels=np.unique(labels))
    return {
        "acc": acc,
        "macro_f1": f1,
        "macro_recall":r,
        "macro_precision": p
    }


if __name__ == '__main__':
    with open('../final_data/CHEF/test.json', 'r') as f:
        eval_data = json.load(f)
        all_labels = [data['label'] for data in eval_data]

    now_path = '../test_result/stage2/Qwen2_stage2_train_ep4_rank128_lr3e-4/beam1/EGP_results.json'
    print(now_path)
    with open(now_path, 'r') as f:
        pred_dict = json.load(f)

    pred_list = [pred_dict[str(i)] for i in range(len(eval_data))]
    all_preds = [parserv2(text) for text in pred_list]

    score_dic = compute_metrics_fn(all_preds, all_labels)
    print(f"acc = {score_dic['acc']},  macro_f1 = {score_dic['macro_f1']}, macro_recall = {score_dic['macro_recall']}, macro_precision = {score_dic['macro_precision']}")


