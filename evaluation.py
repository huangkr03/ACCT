import os
import json
from pprint import pprint
import re
import argparse
import sys
import requests
from tqdm import tqdm
from utils import extract_code_block

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
        code_block = extract_code_block(item['model_output'])
        if 'model_output' in item and 'task_id' in item:
            transformed_data.append({
                'task_id': item['task_id'],
                'completion': code_block,
            })
    return transformed_data

def transform_taco_data(data):
    """
    Transforms the data into a format suitable for taco evaluation.
    """
    transformed_data = []
    
    for item in data:
        code_block = extract_code_block(item['model_output'])
        if 'model_output' in item and 'task_id' in item:
            transformed_data.append({
                'task_id': item['task_id'],
                'prompt': item['question'],
                'output': [code_block],
            })
    return transformed_data

def transform_concode_data(data):
    transformed_data = []
    answer_data = []
    
    for item in data:
        code_block = extract_code_block(item['model_output'], language='java').strip().removeprefix("Solution:").removeprefix("Answer:").lstrip()
        code_block = code_block.split(" //")[0].strip()
        if 'model_output' in item:
            answer_data.append({
                'nl': item['nl'],
                'code': item['answer'],
            })
            transformed_data.append({
                'nl': item['nl'],
                'code': code_block,
            })
    return transformed_data, answer_data

def transform_conala_data(data):
    transformed_data = []
    answer_data = []
    
    for item in data:
        code_block = extract_code_block(item['model_output'], language='python').strip().removeprefix("Solution:").removeprefix("Answer:").lstrip()
        if 'model_output' in item:
            answer_data.append({
                'nl': item['nl'],
                'code': item['answer'],
            })
            transformed_data.append({
                'nl': item['nl'],
                'code': code_block,
            })
    return transformed_data, answer_data

def evaluate_humaneval(args):
    predictions = transform_human_eval_data(load_jsonl(os.path.join(args.input_dir, args.pred_file)))
    data_path = os.path.join(args.input_dir, 'transformed_' + args.pred_file)
    with open(data_path, 'w', encoding='utf-8') as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    os.system(f'evaluate_functional_correctness {data_path}')

def evaluate_concode(args):
    predictions, answers = transform_concode_data(load_jsonl(os.path.join(args.input_dir, args.pred_file)))
    data_path = os.path.join(args.input_dir, 'predictions.txt')
    answer_path = os.path.join(args.input_dir, 'answers.jsonl')
    with open(data_path, 'w', encoding='utf-8') as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(answer_path, 'w', encoding='utf-8') as f:
        for item in answers:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    os.system(f'python evaluator/evaluator.py -p={data_path} -a={answer_path}')
    
def evaluate_conala(args):
    predictions, answers = transform_conala_data(load_jsonl(os.path.join(args.input_dir, args.pred_file)))
    data_path = os.path.join(args.input_dir, 'predictions.txt')
    answer_path = os.path.join(args.input_dir, 'answers.jsonl')
    with open(data_path, 'w', encoding='utf-8') as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(answer_path, 'w', encoding='utf-8') as f:
        for item in answers:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    os.system(f'python evaluator/evaluator.py -p={data_path} -a={answer_path}')
    
def evaluate_codeforces(args):
    predictions = load_jsonl(os.path.join(args.input_dir, args.pred_file))
    # for item in predictions:
    for item in tqdm(predictions, desc="Evaluating Codeforces predictions"):
        if item['finish_reason'] != 'stop':
            continue
        code_block = extract_code_block(item['model_output'], language='python').strip()
        if not code_block:
            continue
        source_code = code_block
        endpoint = "http://localhost:2000/api/v2"
        extension, piston_language = "py", "python3"
        # extension, piston_language = "cpp", "cf_c++17"
        # problem_data is a row from this dataset
        test_case = item['official_tests'][0]  # if this problem also has generated_tests, you should run those too
        # print(source_code)
        payload = {
            "language": piston_language,
            "version": "*", 
            "files": [
                {
                    "name": f"main.{extension}",
                    "content": source_code
                },
                {
                    "name": "input.txt",
                    "content": test_case['input']
                },
                {
                    "name": "correct_output.txt", 
                    "content": test_case['output']
                },
                *([{"name": "checker.py", "content": item['generated_checker']}] if item['generated_checker'] else []),
                {
                    "name": "grader_config",
                    "content": "\n".join(
                        f"{key}={value}" for key, value in {
                            "TIME_LIMIT": item['time_limit'],
                            "MEMORY_LIMIT": item['memory_limit'],
                            "INPUT_MODE": item['input_mode'] if 'input_mode' in item else "stdio",
                        }.items()
                    )
                }
            ]
        }
        result = requests.post(f"{endpoint}/execute", json=payload, headers={"Content-Type": "application/json"})
        json_result = result.json()
        item['eval_result'] = json_result
    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='human_eval', help='The benchmark to process.')
    parser.add_argument('--input_dir', type=str, default='outputs/DeepSeek-R1-Distill-Qwen-7B/sft_adaptive/human_eval/7b/test/samples', help='The input directory.')
    parser.add_argument('--pred_file', type=str, default='predictions.jsonl', help='The prediction file.')
    
    args = parser.parse_args()
    
    if args.benchmark == 'human_eval':
        evaluate_humaneval(args)
    elif args.benchmark == 'taco':
        data = transform_taco_data(load_jsonl(os.path.join(args.input_dir, args.pred_file)))
    elif args.benchmark == 'concode':
        evaluate_concode(args)
    elif args.benchmark == 'conala':
        evaluate_conala(args)
    elif args.benchmark == 'codeforces':
        predictions = evaluate_codeforces(args)
        # with open(os.path.join(args.input_dir, 'eval_results.jsonl'), 'w', encoding='utf-8') as f:
        #     save_jsonl(predictions, f)
        save_jsonl(predictions, os.path.join(args.input_dir, 'eval_results.jsonl'))
        
        # sys.exit(0)
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
    
    # if args.benchmark == 'taco':
    #     with open('generation.json', 'w', encoding='utf-8') as f:
    #         json.dump(data, f, ensure_ascii=False)
    
    if args.benchmark == 'human_eval':
        results = load_jsonl(f'{args.input_dir}/results.jsonl')
    elif args.benchmark == 'codeforces':
        results = []
        for pred in predictions:
            temp = {}
            if 'eval_result' in pred:
                result = pred['eval_result']
                if result and 'compile' in result and result['compile']['code'] == 0 and result['run']['code'] == 0 and result['run']['stdout'].strip() == '1':
                    temp['passed'] = True
                else:
                    temp['passed'] = False
            else:
                temp['passed'] = False
            results.append(temp)
    else:
        results = []
        sys.exit(0)
    
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
    pprint(metrics)
    with open(f'{args.input_dir}/metrics.json', 'w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=4)
        
    
