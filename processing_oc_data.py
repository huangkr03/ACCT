# process opencompass data
import json
import os
from transformers import AutoTokenizer
from tqdm import tqdm

def get_cot_length(data, tokenizer):
    return len(tokenizer(data.split("</think>")[0], return_tensors="pt")['input_ids'][0])
    
def filter_right_answer(data, result_path):
    filtered_data = []
    with open(result_path, 'r') as f:
        result_data = json.load(f)
    results = result_data['details']
    for i, datum in tqdm(enumerate(data)):
        if 'prediction' in data[datum]:
            if 'correct' in results[i] and results[i]['correct']:
                filtered_data.append(data[datum])
            elif 'is_correct' in results[i] and results[i]['is_correct'][0]:
                filtered_data.append(data[datum])
    return filtered_data
        

def filter_cot_length(data, cot_length_filter, tokenizer):
    filtered_data = []
    for datum in tqdm(data):
        if 'prediction' in datum:
            cot = datum['prediction'].split("</think>")[0]
            cot_length = get_cot_length(cot, tokenizer)
            if cot_length < cot_length_filter:
                filtered_data.append(datum)
    return filtered_data

def get_formated_data(data, format="sharegpt"):
    if format == "sharegpt":
        messages = []
        # for prompt in data['origin_prompt']:
        #     # messages.append({'role': prompt['role'], 'content': prompt['prompt']})
        #     messages.append({'role': "user" if prompt['role'] == "HUMAN" else "assistant", 'content': prompt['prompt']})
        messages.append({'role': "user" if data['origin_prompt'][-1]['role'] == "HUMAN" else "assistant", 'content': data['origin_prompt'][-1]['prompt']})
        # messages.append({'role': 'BOT', 'content': data['prediction']})
        messages.append({'role': 'assistant', 'content': data['prediction']})
        formated_data = {
            'messages': messages,
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

def process_opencompass_data(data_path, result_path, data_name="dataset-gsm8k-oc.json", cot_length_filter=1024):
    tokenizer = AutoTokenizer.from_pretrained("../models/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
    if not os.path.exists(data_path):
        print(f"File {data_path} does not exist.")
        return
    with open(data_path, 'r') as f:
        data = json.load(f)
    right_answer_data = filter_right_answer(data, result_path)
    filtered_data = filter_cot_length(right_answer_data, cot_length_filter, tokenizer)
    formated_data = [get_formated_data(data) for data in filtered_data]
    
    save_to_llamafactory(formated_data, data_name)
    
    print(f"Filtered data saved to {data_name} with {len(filtered_data)} samples.")
    
# def process_opencompass_data_v2(data_paths, result_paths, data_name="dataset-gsm8k-oc.json", cot_length_filter=1024):
#     """ BoN """
#     tokenizer = AutoTokenizer.from_pretrained("../models/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
#     if 
    
    
if __name__ == "__main__":
    cot_length_filter = 1024
    
    # data_path = "~/github/opencompass/outputs/default/gsm8k-train/predictions/deepseek-7b-chat-vllm/gsm8k.json"
    # result_path = "~/github/opencompass/outputs/default/gsm8k-train/results/deepseek-7b-chat-vllm/gsm8k.json"
    # process_opencompass_data(data_path, result_path, cot_length_filter=cot_length_filter)
    
    data_path = "~/github/opencompass/outputs/default/mbpp-train/predictions/deepseek-7b-chat-vllm/mbpp.json"
    result_path = "~/github/opencompass/outputs/default/mbpp-train/results/deepseek-7b-chat-vllm/mbpp.json"
    process_opencompass_data(data_path, result_path, data_name="dataset-mbpp.json", cot_length_filter=cot_length_filter)