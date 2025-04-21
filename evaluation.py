import os
import json
import re
from tqdm import tqdm
import argparse

def load_jsonl(file, encoding='utf-8'):
    data = []
    with open(file, 'r', encoding=encoding) as f:
        for j in f.readlines():
            j = json.loads(j)
            data.append(j)
    return data

def save_jsonl(data, file, encoding='utf-8'):
    with open(file, 'w', encoding=encoding) as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def transform_human_eval_data(data):
    """
    Transforms the data into a format suitable for human_eval evaluation.
    """
    transformed_data = []
    
    for item in data:
        # get code from the last ```python\n ```
        regex = r'```python\n(.*?)\n```'
        code_pieces = re.findall(regex, item['model_output'], re.DOTALL)
        if code_pieces:
            code_piece = code_pieces[-1]  # 返回最后一个匹配项
        else:
            code_piece = ''
        if 'model_output' in item and 'task_id' in item:
            transformed_data.append({
                'task_id': item['task_id'],
                'completion': code_piece,
            })
    return transformed_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='human_eval', help='The benchmark to process.')
    parser.add_argument('--input_dir', type=str, default='outputs/DeepSeek-R1-Distill-Qwen-7B/sft_adaptive/human_eval/7b/test/samples', help='The input directory.')
    parser.add_argument('--pred_file', type=str, default='predictions.jsonl', help='The prediction file.')
    
    args = parser.parse_args()
    
    data = transform_human_eval_data(load_jsonl(os.path.join(args.input_dir, args.pred_file)))
    
    save_path = os.path.join(args.input_dir, 'transformed_' + args.pred_file)
    save_jsonl(data, save_path)
    print(f"Transformed data saved to {save_path}")
    
    os.system(f'evaluate_functional_correctness {save_path}')
    
    results = load_jsonl(f'{save_path}_results.jsonl')
    
    total = len(results)
    passed = 0
    for item in results:
        if item['passed']:
            passed += 1
    
    acc = passed / total
    
    metrics = {}
    if os.path.exists(f'{args.input_dir}/metrics.json'):
        with open(f'{args.input_dir}/metrics.json', 'r', encoding='utf-8') as metrics_file:
            metrics = json.load(metrics_file)
    metrics['accuracy'] = acc
    with open(f'{args.input_dir}/metrics.json', 'w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=4)
        
    
