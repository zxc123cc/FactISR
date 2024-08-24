import json
all_data = []
merged_dict = {}

for fold in range(5):
    path = f'../test_result/stage1/Qwen2_stage1_train_ep4_trending/EGP_fold{fold}.json'
    with open(path, 'r') as f:
        fold_data = json.load(f)
        merged_dict.update(fold_data)

with open('../test_result/stage1/Qwen2_stage1_train_ep4_trending/EGP_train_all.json', 'w') as f:
    json.dump(merged_dict, f)