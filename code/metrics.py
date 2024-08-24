import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BertTokenizer, BertModel
from bert_score import score
from rouge import Rouge

import torch
import bert_score


# 加载本地BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 将模型移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def calculate_bert_score(sent1, sent2, model_path='bert-base-chinese'):
    # 最大序列长度
    max_length = 512
    
    # 手动对输入句子进行分词、填充和截断
    inputs_sent1 = tokenizer(sent1, truncation=True, max_length=max_length, padding="max_length", return_tensors='pt')
    inputs_sent2 = tokenizer(sent2, truncation=True, max_length=max_length, padding="max_length", return_tensors='pt')
    
    # 将 tokenized 序列解码回字符串形式
    processed_sent1 = tokenizer.decode(inputs_sent1['input_ids'][0], skip_special_tokens=True)
    processed_sent2 = tokenizer.decode(inputs_sent2['input_ids'][0], skip_special_tokens=True)
    
    # 使用 bert-score 库计算 BERTScore
    P, R, F1 = bert_score.score([processed_sent1], [processed_sent2], model_type=model_path, lang="zh", verbose=True)
    
    return F1.item()


def cal_one_sample_score(hypothesis, reference, bert_model_type):
    # 使用jieba进行中文分词
    hypothesis_tokens = list(jieba.cut(hypothesis))
    reference_tokens = list(jieba.cut(reference))

    # 计算BLEU-4
    smooth_func = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smooth_func)

    # 计算BERTScore
    bert_score = calculate_bert_score(hypothesis, reference)

    # 计算ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(" ".join(hypothesis_tokens), " ".join(reference_tokens))[0]

    return {
        "BLEU-4": bleu_score,
        "BERTScore": bert_score,
        "ROUGE-1": rouge_scores['rouge-1'],
        "ROUGE-2": rouge_scores['rouge-2'],
        "ROUGE-L": rouge_scores['rouge-l']
    }

def evaluate_metrics(hypothesis_list,reference_list,bert_model_type):
    metrics = {
        "BLEU-4": [],
        "BERTScore": [],
        "ROUGE-1": [],
        "ROUGE-2": [],
        "ROUGE-L": []
    }
    for hypothesis, reference in zip(hypothesis_list,reference_list):
        if hypothesis=='':
            continue
        try:
            now_metrics = cal_one_sample_score(hypothesis, reference,bert_model_type=bert_model_type)
            metrics['BLEU-4'].append(now_metrics['BLEU-4'])
            metrics['BERTScore'].append(now_metrics['BERTScore'])
            metrics['ROUGE-1'].append(now_metrics['ROUGE-1']['f'])
            metrics['ROUGE-2'].append(now_metrics['ROUGE-2']['f'])
            metrics['ROUGE-L'].append(now_metrics['ROUGE-L']['f'])
        except:
            continue
    
    metrics['BLEU-4'] = sum(metrics['BLEU-4']) / len(metrics['BLEU-4'])
    metrics['BERTScore'] = sum(metrics['BERTScore']) / len(metrics['BERTScore'])
    metrics['ROUGE-1'] = sum(metrics['ROUGE-1']) / len(metrics['ROUGE-1'])
    metrics['ROUGE-2'] = sum(metrics['ROUGE-2']) / len(metrics['ROUGE-2'])
    metrics['ROUGE-L'] = sum(metrics['ROUGE-L']) / len(metrics['ROUGE-L'])
    
    return metrics
    
    

# # 示例文本
# hypothesis = "北京是中国的首都，拥有丰富的历史和文化。"
# reference = "中国的首都是北京，它有着悠久的历史和文化。"

# # 计算各项指标
# metrics = evaluate_metrics(hypothesis, reference,bert_model_type='/home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/zhangxiaocheng/opensource_model_weights/PollyZhao/bert-base-chinese')
# print("BLEU-4 Score:", metrics['BLEU-4'])
# # print("BERTScore:", metrics['BERTScore'])
# print("ROUGE-1:", metrics['ROUGE-1'])
# print("ROUGE-2:", metrics['ROUGE-2'])
# print("ROUGE-L:", metrics['ROUGE-L'])
