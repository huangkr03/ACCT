import os
import json
from pprint import pprint
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from copy import deepcopy
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch
from time import time
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer
import re
from utils import dataloader

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_data(path):
    if path.endswith("json"):
        data = json.load(open(path, "r"))
    elif path.endswith("jsonl"):
        data = []
        with open(path, "r") as file:
            for line in file:
                line = json.loads(line)
                data.append(line)
    else:
        raise NotImplementedError()
    return data

def extract_boxed_answer(text):
    start_index = 0
    # 查找下一个 \boxed{ 的起始位置
    try:
        # 使用 find 方法查找 '\boxed{'
        # 加 5 是为了跳过 '\boxed{' 本身，直接定位到内容的开始
        match_start = text.index('\\boxed{', start_index) + 7
    except ValueError:
        # 如果找不到 '\boxed{'，结束查找
        return ""
    level = 1  # 括号层级，初始为 1 因为已经匹配了 \boxed{ 的 {
    content_start = match_start
    current_index = match_start
    # 寻找匹配的结束括号 }
    while current_index < len(text):
        char = text[current_index]
        if char == '{':
            level += 1
        elif char == '}':
            level -= 1
            if level == 0:
                # 找到了匹配的结束括号
                # 提取从 content_start 到当前位置的内容
                # contents.append(text[content_start:current_index])
                return text[content_start:current_index]
        current_index += 1
    else:
        # 如果循环结束还没有找到匹配的括号（level != 0）
        # 说明括号不匹配或者字符串在找到匹配括号前就结束了
        # 为了避免无限循环，我们需要跳出主循环或根据需要处理错误
        # 这里选择跳出主循环
        print(f"警告：在索引 {content_start-7} 处发现未闭合的 \\boxed{{}}")
        return ""

def compare_answer(pred, ans, prec=1e-3):
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False
    if isinstance(pred, list):
        pred = ','.join(pred)
    if isinstance(ans, list):
        ans = ','.join(ans)
    # Remove commas, spaces, and special characters for comparison
    for char in [',', ' ', '%', '$', '\\', ':', ';', '\n', '\t', '\r', '(', ')', '[', ']', '!']:
        pred = pred.replace(char, "")
        ans = ans.replace(char, "")
    
    # Convert to lowercase for case-insensitive comparison
    pred = pred.lower()
    ans = ans.lower()
    
    
    if pred == ans:
        return True
    if is_float(ans):
        if not is_float(pred):
            pred = re.sub(r'[^\d.]', '', pred).strip()
            
        if is_float(pred):
            try:
                return abs(float(re.sub(r',', '', str(pred))) - float(re.sub(r',', '', str(ans)))) < prec
            except:
                return pred == ans
    return pred == ans

def compare_original(original_path, new_path):
    original_answers = read_data(original_path)
    new_answers = read_data(new_path)

    # Filter out the answers with finish_reason != "stop"
    filtered_original_answers = []
    filtered_new_answers = []
    for i in range(len(original_answers)):
        if original_answers[i]['finish_reason'] == 'stop' and new_answers[i]['finish_reason'] == 'stop':
            filtered_original_answers.append(original_answers[i])
            filtered_new_answers.append(new_answers[i])

    # Calculate the correctness rate
    succ_original = len(list(filter(lambda x: x['correctness'] == 1, filtered_original_answers)))
    succ_new = len(list(filter(lambda x: x['correctness'] == 1, filtered_new_answers)))

    original = f'Original: {succ_original / len(filtered_original_answers):.2%}'
    new = f'New: {succ_new / len(filtered_new_answers):.2%}'
    print(f'Original: {succ_original / len(filtered_original_answers):.2%}')
    print(f'New: {succ_new / len(filtered_new_answers):.2%}')
    return original, new

def evaluate_answer_llm(llm, sampling_params, outputs, answers):
    def get_correctness(output):
        if 'yes' in output.lower():
            return 1
        return 0
    messages = []
    for output, ans in zip(outputs, answers):
        message = [{
            'role': 'user',
            'content': f"Please judge the correctness of the output with the right answer:\n\nModel Output: \n{output}\nRight Answer: {ans}\n\nIs the model output correct? Please simply response with 'yes' or 'no'. \
                PS: Just compare the output of the model with the correct answers, ignore the format. For instance, if the answer is \\frac{1}{2}, then 0.5 or 1/2 would also be considered correct. Moreover, if the right answer is '10', then '10' or '10%' could be considered correct, due to mathematical unit issues, the standard answer may not include units, while the output does. In cases where multiple answers are required, providing an incomplete answer will be marked as incorrect."
        }]
        messages.append(message)
    
    llm_outputs = llm.chat(messages=messages, sampling_params=sampling_params)
    correctnesses = [get_correctness(llm_output.outputs[0].text) for llm_output in llm_outputs]
    os.makedirs('outputs/temp', exist_ok=True)
    with open(f'{output_dir}/llm_eval.jsonl', 'w') as f:
        for i, (llm_output, output) in enumerate(zip(llm_outputs, outputs)):
            f.write(json.dumps({
                'eval_output': llm_output.outputs[0].text,
                'right_answer': answers[i],
                'model_output': output,
                'correctness': get_correctness(llm_output.outputs[0].text),
                                }, ensure_ascii=False) + '\n')
    return correctnesses
    

def infer(args, test_data):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    messages = []
    
    if args.benchmark == "human_eval":
        # 处理 human_eval 数据集
        for example in test_data:
            prompt = example['prompt']
            message = [{
                'role': 'user',
                'content': prompt
            }]
            messages.append(message)
    elif args.benchmark == "taco":
        # 处理 taco 数据集
        for example in test_data:
            problem = example['question']
            message = [{
                'role': 'user',
                'content': problem
            }]
            messages.append(message)
    elif args.benchmark == "concode" or args.benchmark == "conala":
        for example in test_data:
            problem = example['prompt']
            message = [{
                'role': 'user',
                'content': problem
            }]
            messages.append(message)
    elif args.benchmark == "codeforces":
        # 处理 codeforces 数据集
        for example in test_data:
            problem = example['prompt']
            message = [{
                'role': 'user',
                'content': problem
            }]
            messages.append(message)
    else:
        for example in test_data:
            problem = example['problem'] if 'problem' in example else example['question']
            message = [{
                'role': 'user',
                'content': "Please reason step by step, and put your final answer within \\boxed{}.\n" + problem
            }]
            messages.append(message)
    

    print("加载模型和分词器...")
    print(f"\033[91mprompt demo: {messages[0]}\033[0m")
    
    # 配置VLLM的停止序列
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    
    # 配置VLLM采样参数
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=stop_token_ids
    )
    
    # 初始化VLLM模型
    print("初始化VLLM模型...")
    
    print(f'\033[91m模型路径: {args.model_path}\033[0m')
    model_kwargs = {
        "model": args.model_path,
        "tokenizer": args.tokenizer_path,
        "tensor_parallel_size": args.num_gpus,
        "enable_lora": args.use_adapter,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_lora_rank": 16,
        # "max_model_len": args.max_new_tokens + 2048
        "max_model_len": args.max_new_tokens
    }
    
    llm = LLM(**model_kwargs)
    
    # 使用VLLM批量生成
    torch.cuda.synchronize()
    start_time = time()
    
    final_results = []
    for i in range(args.num_per_example):
        if args.use_adapter:
            print(f"\033[91muse adapter: {args.adapter_path}\033[0m")
        if args.use_adapter:
            outputs = llm.chat(messages=messages, sampling_params=sampling_params, lora_request=LoRARequest(lora_name="lora_sft", lora_int_id=1, lora_path=args.adapter_path))
        else:
            outputs = llm.chat(messages=messages, sampling_params=sampling_params)
        torch.cuda.synchronize()
        total_time = time() - start_time
        print(f"生成完成，耗时: {total_time:.2f}秒")

        print(f'\033[91m模型输出:\033[0m')
        print(outputs[0])
        print(outputs[0].outputs[0].text)
        print(f'\033[91m{outputs[0].finished}\033[0m')
        print(f'\033[91m{outputs[0].outputs[0].finish_reason}\033[0m')

        # 提取生成的文本
        model_outputs = [output.outputs[0].text for output in outputs]
        finish_reasons = [output.outputs[0].finish_reason for output in outputs]

        # 计算思维链长度
        cot_lengths = []
        for model_completion in model_outputs:
            cot = model_completion.split('</think>')[0]
            cot_length = tokenizer(cot, return_tensors="pt")['input_ids'].shape[1]
            cot_lengths.append(cot_length)
    
        # 提取预测答案
        # predictions = [extract_boxed_answer(output) for item, output in tqdm(zip(test_data, model_outputs), desc="提取答案", total=len(model_outputs))]
        # assert len(model_outputs) > 0, f"没有生成输出: {len(model_outputs)}"
        if len(model_outputs) == 0:
            print(f"没有生成输出: {len(model_outputs)}")
            continue
            

        full_correctnesses = []
        # 提取预测答案
        if args.benchmark != "human_eval" and args.benchmark != "taco" and args.benchmark != "concode" and args.benchmark != "conala" and args.benchmark != "codeforces":
            boxed_answers = []
            # filter finish_reason = "stop" (both model_outputs and answer)
            output_extracted, answer_filtered = [], []
            for i, finish_reason in enumerate(finish_reasons):
                if finish_reason == "stop":
                    extracted_answer = extract_boxed_answer(model_outputs[i])
                    if extracted_answer != "":
                        boxed_answers.append(1)
                        output_extracted.append(extracted_answer)
                        answer_filtered.append(test_data[i]['answer'])
                        continue
                boxed_answers.append(0)
            
            correctnesses = evaluate_answer_llm(llm, sampling_params, output_extracted, answer_filtered)
            print(f"\033[91mcorrectnesses: {len(correctnesses)}\033[0m")
            print(f"\033[91mcorrectnesses: {len(output_extracted)}\033[0m")
        
            # Create a correctness list matching the size of finish_reasons
            correctness_idx = 0

            for i in range(len(finish_reasons)):
                if boxed_answers[i] == 1:
                    full_correctnesses.append(correctnesses[correctness_idx])
                    correctness_idx += 1
                else:
                    full_correctnesses.append(-1)
        else:
            boxed_answers = [''] * len(finish_reasons)
            full_correctnesses = [0] * len(finish_reasons)
        # 组装结果
        results = []
        for i, (example, output, cot_length, finish_reason, formatted_answer) in enumerate(zip(test_data, model_outputs, cot_lengths, finish_reasons, boxed_answers)):
            item = deepcopy(example)
            item.update({
                'model_output': output,
                'cot_length': cot_length,
                'finish_reason': finish_reason,
                'correctness': full_correctnesses[i],
                'formatted_answer': formatted_answer,
            })
            results.append(item)
        
        n_change = 0
        if final_results:
            for i, new_result in enumerate(results):
                if i < len(final_results):
                    if final_results[i]['correctness'] != 1 and new_result['correctness'] == 1:
                        final_results[i] = new_result
                        n_change += 1
                    if "</think>" not in final_results[i]['model_output'] and "</think>" in new_result['model_output']:
                        final_results[i] = new_result
                        n_change += 1
                    if final_results[i]['correctness'] == 1 and new_result['correctness'] == 1:
                        if final_results[i]['cot_length'] > new_result['cot_length']:
                            final_results[i] = new_result
                            n_change += 1
            print(f'\033[91m更新模型输出: {n_change}个\033[0m')
        else:
            final_results = results
        
    return final_results, total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/DeepSeek-R1-Distill-Qwen-7B/", help="default to `model_path`_predictions")
    parser.add_argument("--model_path", type=str, default="../models/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default="../models/DeepSeek-R1-Distill-Qwen-7B/lora_sft")
    parser.add_argument("--model_size", type=str, default="7b")
    parser.add_argument("--use_adapter", action='store_true', default=False, help="whether to use LoRA")
    parser.add_argument("--benchmark", type=str, choices=['r1', 'gsm8k', 'math', 'human_eval', 'taco', 'concode', 'conala', 'codeforces'], default="gsm8k")
    parser.add_argument("--data_type", type=str, choices=['train', 'test'], default="test")

    parser.add_argument("--max_num_examples", type=int, default=100000000, help="maximum number of examples to evaluate.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--num_gpus", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--num_per_example", type=int, default=1, help="number of answers per example (only choose the best one)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--origin_path", type=str, default=None, help="path to the original data eval path")
    
    parser.add_argument("--save_period", type=int, default=50, help="save every N steps")
    
    args, unparsed_args = parser.parse_known_args()
    
    if args.model_path and not args.tokenizer_path:
        args.tokenizer_path = args.model_path

    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    print(f"Evaluating {args.model_path}", flush=True)
    print(f"Max new tokens: {args.max_new_tokens}, temperature: {args.temperature}, seed: {args.seed}\n", flush=True)
    if args.use_adapter:
        print(f"Adapter path {args.adapter_path}", flush=True)

    if args.use_adapter:
        args.output_dir = os.path.join(args.output_dir, f"{args.benchmark}", f"{args.model_size}/", f'{args.adapter_path.split("/")[-1]}_{args.data_type}/')
    else:
        args.output_dir = os.path.join(args.output_dir, f"{args.benchmark}", f"{args.model_size}/", f"{args.data_type}/")

    dataset_path = ''
    if args.benchmark == "r1":
        dataset_path = "datasets/OpenR1_Math_220k_test.json"
    elif args.benchmark == "human_eval":
        dataset_path = f"datasets/human_eval.json"
    elif args.benchmark == "gsm8k":
        dataset_path = f"datasets/gsm8k/{args.data_type}.jsonl"
    elif args.benchmark == "math":
        dataset_path = f"datasets/math-500/{args.data_type}.jsonl"
    elif args.benchmark == "taco":
        dataset_path = f"datasets/TACO/{args.data_type}.jsonl"
    if dataset_path:
        test_data = read_data(dataset_path)
    elif args.benchmark == "concode":
        concode = dataloader.Concode()
        test_data = concode.get_dataset(train=args.data_type == "train")
    elif args.benchmark == "conala":
        conala = dataloader.Conala()
        test_data = conala.get_dataset(train=args.data_type == "train")
    elif args.benchmark == "codeforces":
        dataset_path = f"datasets/codeforces/{args.data_type}_prompts.jsonl"
    else:
        raise NotImplementedError(f"Unsupported benchmark: {args.benchmark}")
    
    if dataset_path:
        test_data = read_data(dataset_path)
    
    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)
        
    print(test_data[0].keys())
    
    print(f"Loaded {len(test_data)} examples from {dataset_path}", flush=True)
    
    output_dir = os.path.join(args.output_dir, "samples")
    
    os.makedirs(output_dir, exist_ok=True)

    results, total_time = infer(args, test_data)

    print("Finished inference...")

    # os.environ['TOKENIZERS_PARALLELISM'] = "false"

    invalid_outputs = []
    for item in results:
        if item['finish_reason'] != 'stop':
            invalid_outputs.append({
                'id': item['uuid'] if 'uuid' in item else item.get('id', ''),
                'output': item['model_output'], 
                'answer': ""})

    print("Calculating accuracy...")
    acc = 0
    total = 0
    for item in results:
        acc += 1 if item['correctness'] == 1 else 0
        total += 1 if item['correctness'] != -1 else 0
    if total != 0:
        accruacy = acc / total
    else:
        accruacy = 0
    print("output acc = {:.5f}".format(accruacy), flush=True)
    print("output acc including invalid outputs = {:.5f}".format(acc / len(results)), flush=True)
    
    # unfinished count
    unfinished_count = 0
    for item in results:
        if item['finish_reason'] != 'stop':
            unfinished_count += 1
    print("output unfinished count = {}".format(unfinished_count), flush=True)

    avg_cot_length = sum(item['cot_length'] for item in results) / len(results)
    print("output avg_cot_length = {:.5f}".format(avg_cot_length), flush=True)

    results_finished = [item for item in results if item['finish_reason'] == 'stop']
    if len(results_finished) > 0:
        avg_cot_length_finished = sum(item['cot_length'] for item in results_finished) / len(results_finished)
        print("output avg_cot_length_finished = {:.5f}".format(avg_cot_length_finished), flush=True)
    else:
        avg_cot_length_finished = 0
        print("output avg_cot_length_finished = {:.5f}".format(avg_cot_length_finished), flush=True)
    
    cot_lengths = [item['cot_length'] for item in results]
    middle_cot_length = np.median(cot_lengths)
    print("output middle_cot_length = {:.5f}".format(middle_cot_length), flush=True)

    print("number of invalid outputs: {}".format(len(invalid_outputs)), flush=True)

    pred_fname = "predictions.jsonl"
    
    with open(os.path.join(output_dir, pred_fname), 'w', encoding='utf-8') as fout:
        for item in results:
            line = json.dumps(item, ensure_ascii=False)
            fout.write(line + '\n')
        
        
    if args.origin_path is not None:
        original, new = compare_original(args.origin_path, os.path.join(output_dir, pred_fname))

    metric_fname = "metrics.json"
    with open(os.path.join(output_dir, metric_fname), "w") as fout:
        json_data = {
            "n_samples": len(results),
            "accuracy": accruacy,
            "accuracy_including_invalid_outputs": acc / len(results),
            "max_token_exceeded": len(invalid_outputs),
            "avg_cot_length": avg_cot_length,
            "avg_cot_length_finished": avg_cot_length_finished,
            "middle_cot_length": middle_cot_length,
            'sample_latency': total_time / len(test_data),
            'total_time': total_time,
            'gpu_count': args.num_gpus,
            'gpu_memory_utilization': args.gpu_memory_utilization,
        }
        if args.origin_path is not None:
            json_data['original_accuracy'] = original
            json_data['new_accuracy'] = new
        json.dump(json_data, fout, indent=4)
