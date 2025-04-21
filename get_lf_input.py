import os
import json
import random
import numpy as np


def load_json(file, encoding='utf-8'):
    data = []
    with open(file, 'r', encoding=encoding) as f:
        for j in f.readlines():
            j = json.loads(j)
            data.append(j)
    return data

def write_list_to_json(list, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w') as  f:
        json.dump(list, f, ensure_ascii=False, indent=1)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def load_original_data(input_dir="outputs/DeepSeek-R1-Distill-Qwen-7B/gsm8k/7b/Original/train/samples", file_name="predictions_formatted.jsonl"):
    original_data = load_json(os.path.join(input_dir, file_name))
    return original_data

def copy_to_llamafactory_dir(input_dir="outputs/datasets", 
                             data_name="dataset.json",
                             llamafactory_dir="/home/keruihuang/LLaMA-Factory/data",
                             dataset_name="dataset"):
    """
    Copy the formatted data to the LLaMA-Factory directory.
    Edit dataset_info.json, add the new dataset.
    e.g.
    "dataset": {
        "formatting": "sharegpt",
        "file_name": "dataset.json",
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
    """
    # copy file to llamafactory_dir
    os.system(f"cp {os.path.join(input_dir, data_name)} {llamafactory_dir}/")
    # edit dataset_info.json
    dataset_info_path = os.path.join(llamafactory_dir, "dataset_info.json")
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    dataset_info.update({
        dataset_name: {
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
        
    

def get_llamafactory_input(input_dir="outputs/DeepSeek-R1-Distill-Qwen-7B/gsm8k/7b/Original/train/samples", 
                           file_name="predictions_filtered.jsonl",
                           data_name="dataset.json",
                           llamafactory_dir="/home/keruihuang/LLaMA-Factory/data"):
    original_data = load_original_data(input_dir, file_name)
    datalines = []
    for i in range(len(original_data)):
        # input_data = original_data[i]['messages'][0]['content']
        input_data = original_data[i]['question'] if 'question' in original_data[i] else original_data[i]['problem']
        # answer = original_data[i]['prediction']
        output = original_data[i]['model_output']
        # output_data = f"{cot}\n\nThe final answer is: " + "$\\boxed{" + answer + "}$"
        data = {
            # "instruction": "Please reason step by step, and put your final answer within \\boxed{}.",
            "messages": [
                {
                    "role": "user",
                    "content": "Please reason step by step, and put your final answer within \\boxed{}.\n" + input_data
                },
                {
                    "role": "assistant",
                    "content": output
                }
            ],
        }
        datalines.append(data)
    
    print(len(datalines))
    random.shuffle(datalines)
    write_list_to_json(datalines, f'./outputs/datasets/{data_name}')
    copy_to_llamafactory_dir(input_dir="./outputs/datasets",
                             data_name=data_name,
                             llamafactory_dir=llamafactory_dir,
                             dataset_name=data_name.rstrip(".json"))


if __name__ == '__main__':
    data_name = "dataset-gsm8k-all.json"  # change as you like
    llamafactory_dir = "/home/keruihuang/LLaMA-Factory/data"  # llama factory data dir
    # input_dir = "outputs/DeepSeek-R1-Distill-Qwen-32B/train_1024_5/gsm8k/7b/train/samples"  # dir that contains predictions_filtered.jsonl
    input_dir = "/home/keruihuang/cot_compression/outputs/DeepSeek-R1-Distill-Qwen-7B/train_8k/gsm8k/7b/train/samples"  # dir that contains predictions_filtered.jsonl
    get_llamafactory_input(input_dir=input_dir, 
                           data_name=data_name,
                           llamafactory_dir=llamafactory_dir)
