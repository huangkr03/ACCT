# process opencompass data
import json
import os
from transformers import AutoTokenizer
from tqdm import tqdm

def get_cot_length(data, tokenizer):
    return len(tokenizer(data.split("</think>")[0], return_tensors="pt")['input_ids'][0])
    
def filter_right_answer(data):
    filtered_data = []
    for datum in tqdm(data):
        if 'eval_result' in datum:
            result = datum['eval_result']
            if result and 'compile' in result and result['compile']['code'] == 0 and result['run']['code'] == 0 and result['run']['stdout'].strip() == '1':
                filtered_data.append(datum)
    
    print(f"Filtered {len(data) - len(filtered_data)} samples with wrong answers.")
    print(f"Kept {len(filtered_data)} samples with right answers.")
    return filtered_data
        

def filter_cot_length(data, cot_length_filter):
    filtered_data = []
    for datum in tqdm(data):
        if 'cot_length' in datum:
            cot_length = datum['cot_length']
            if cot_length < cot_length_filter:
                filtered_data.append(datum)
    print(f"Filtered {len(data) - len(filtered_data)} samples with COT length > {cot_length_filter}.")
    print(f"Kept {len(filtered_data)} samples with COT length < {cot_length_filter}.")
    return filtered_data

def get_formated_data(data, format="sharegpt"):
    if format == "sharegpt":
        # messages = []
        # for prompt in data['origin_prompt']:
        #     # messages.append({'role': prompt['role'], 'content': prompt['prompt']})
        #     messages.append({'role': "user" if prompt['role'] == "HUMAN" else "assistant", 'content': prompt['prompt']})
        # messages.append({'role': "user" if data['origin_prompt'][-1]['role'] == "HUMAN" else "assistant", 'content': data['origin_prompt'][-1]['prompt']})
        # # messages.append({'role': 'BOT', 'content': data['prediction']})
        # messages.append({'role': 'assistant', 'content': data['prediction']})
        # formated_data = {
        #     'messages': messages,
        # }
        formated_data = {
            # "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
            "messages": [
                {
                    "role": "user",
                    "content": data['prompt']
                },
                {
                    "role": "assistant",
                    "content": data['model_output']
                }
            ],
        }
        return formated_data
    return data

def save_to_llamafactory(data, data_name, llamafactory_dir="../LLaMA-Factory/data"):
    with open(os.path.join(llamafactory_dir, data_name), 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    dataset_info_path = os.path.join(llamafactory_dir, "dataset_info.json")
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    dataset_info.update({
        data_name.rsplit('.', 1)[0]: {
            "formatting": "sharegpt",
            "file_name": data_name,
            "columns": {
                "messages": "messages"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
    })
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

def process_opencompass_data(result_path, data_name="data.json", cot_length_filter=8192):
    if not os.path.exists(result_path):
        print(f"File {result_path} does not exist.")
        return
    with open(result_path, 'r') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    right_answer_data = filter_right_answer(data)
    filtered_data = filter_cot_length(right_answer_data, cot_length_filter)
    formated_data = [get_formated_data(data) for data in filtered_data]
    
    save_to_llamafactory(formated_data, data_name)
    
    # print(f"Filtered data saved to {data_name} with {len(filtered_data)} samples.")
    
# def process_opencompass_data_v2(data_paths, result_paths, data_name="dataset-gsm8k-oc.json", cot_length_filter=1024):
#     """ BoN """
#     tokenizer = AutoTokenizer.from_pretrained("../models/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
#     if 
    
    
if __name__ == "__main__":
    cot_length_filter = 8192
    
    # data_path = "~/github/opencompass/outputs/default/gsm8k-train/predictions/deepseek-7b-chat-vllm/gsm8k.json"
    # result_path = "~/github/opencompass/outputs/default/gsm8k-train/results/deepseek-7b-chat-vllm/gsm8k.json"
    # process_opencompass_data(data_path, result_path, cot_length_filter=cot_length_filter)
    
    result_path = "outputs/DeepSeek-R1-Distill-Qwen-7B/baseline-16k/codeforces/7b/train/samples/eval_results.jsonl"
    process_opencompass_data(result_path, data_name="codeforces-16k.json", cot_length_filter=cot_length_filter)