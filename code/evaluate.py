import json
from metrics import evaluate_metrics


with open('../final_data/CHEF/test.json','r') as f:
    eval_data = json.load(f)
    # eval_data = [data['human_revision'] if data['human_revision'] !="" else data['explanation'] for data in eval_data]
    eval_data = [data['human_revision'] for data in eval_data]
    
    
# with open('../final_data/CHEF/test.json','r') as f:
#     pred_data = json.load(f)
#     # eval_data = [data['human_revision'] if data['human_revision'] !="" else data['explanation'] for data in eval_data]
#     pred_list = [data['llm_explanation'] for data in pred_data]

now_path = '../test_result/stage2/Qwen2_stage2_train_ep3_rank128_lr3e-4/beam1/SCP_results_round1.json'

with open(now_path,'r') as f:
    pred_dict = json.load(f)
    
pred_list = [pred_dict[str(i)] for i in range(len(eval_data))]

metrics = evaluate_metrics(pred_list, eval_data,
                           bert_model_type='bert-base-chinese')

print(now_path)
print(metrics)
