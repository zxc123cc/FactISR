import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_list_to_str(evidence):
    evidence_list = [v['text'] if 'text' in v else '' for k, v in evidence.items()][:2]
    evidence_list = [f'"证据{i + 1}: {evidence}"' for i, evidence in enumerate(evidence_list)]
    now_str = '[' + ','.join(evidence_list) + ']'
    return now_str


# input_template
USER_INPUT_TEMPLATE = '''你是一个事实核查领域解释生成的专家，擅长根据证据和真实性生成相应的解释。"
"我将提供当前说法、相应的证据以及当前说法真实性，你要根据证据和真实性，生成一段解释，说明当前说法为什么是对的/错的/证据不足或不充分的。"
"注意，你的解释要严格按照证据和真实性来。"
"当前说法为：{}"
"对应证据为：{}"
"请你生成解释：
'''

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("/mnt/user/luyifei/model_weight/glm4_trending_lora_sft",trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "/mnt/user/luyifei/model_weight/glm4_trending_lora_sft",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()
# Trending_Data
with open('/mnt/user/luyifei/cs_data/final_data/final_data/Trending/test.json' ,'r') as f:
    Treding_test = json.load(f)
res = {}
for index in tqdm(range(len(Treding_test))):
    claim = Treding_test[index]['claim']
    if 'text' not in Treding_test[index]['evidence']:
        evidence_list = Treding_test[index]['evidence']
        evidence = convert_list_to_str(evidence_list)
    else:
        evidence = Treding_test[index]['evidence']['text']
    query = USER_INPUT_TEMPLATE.format(claim, evidence)
    print(query)
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                        add_generation_prompt=True,
                                        tokenize=True,
                                        return_tensors="pt",
                                        return_dict=True
                                        )

    inputs = inputs.to(device)
    gen_kwargs = {"max_length": 5000, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    res[str(index)] = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # break
with open('/mnt/user/luyifei/cs_data/Trending_dpo_infer.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)