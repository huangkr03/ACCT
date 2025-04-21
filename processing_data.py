import os
import json

def load_jsonl(file, encoding='utf-8'):
    data = []
    with open(file, 'r', encoding=encoding) as f:
        for j in f.readlines():
            j = json.loads(j)
            data.append(j)
    return data

def save_jsonl(data, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    for item in data:
        with open(output_path, 'a+', encoding='utf-8') as f:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')

def filter_correct_outputs(input_path, output_path):
    """
    Filter the correct outputs from the data.
    """
    data = load_jsonl(input_path)
    correct_data = []
    for i in range(len(data)):
        if data[i]['correctness'] and "</think>" in data[i]['model_output'] and data[i]['finish_reason'] == "stop":
            correct_data.append(data[i])
    print(f"Original Samples: {len(data)}, Correct Samples: {len(correct_data)}, Accuracy: {len(correct_data) / len(data)}")
    save_jsonl(correct_data, output_path)


def filter_cot_length(input_path, output_path):
    """
    Filter the formatted outputs from the data. Extract COT from th outputs.
    """
    data = load_jsonl(input_path)
    formatted_data = []
    for i in range(len(data)):
        if data[i]['cot_length'] > max_length:
            continue
        formatted_data.append(data[i])
    print(f"Original Samples: {len(data)}, Formatted Samples: {len(formatted_data)}")
    save_jsonl(formatted_data, output_path)

def data_processing(input_dir="outputs/DeepSeek-R1-Distill-Qwen-7B/gsm8k/7b/Original/train/samples", pred_file="predictions.jsonl"):
    """
    The overall pipeline to process the Math data.
    """
    input_path = os.path.join(input_dir, pred_file)
    correct_path = os.path.join(input_dir, "predictions_correct.jsonl")
    filtered_path = os.path.join(input_dir, "predictions_filtered.jsonl")

    filter_correct_outputs(input_path=input_path, output_path=correct_path)
    filter_cot_length(input_path=correct_path, output_path=filtered_path)

if __name__ == '__main__':
    max_length = 8192
    # input_dir = "outputs/DeepSeek-R1-Distill-Qwen-32B/train_512_1/gsm8k/7b/train/samples"
    input_dir = "/home/keruihuang/cot_compression/outputs/DeepSeek-R1-Distill-Qwen-7B/train_8k/gsm8k/7b/train/samples"
    pred_file = "predictions.jsonl"
    data_processing(input_dir=input_dir, pred_file=pred_file)


