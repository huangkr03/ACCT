

import json
from pprint import pprint


system = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

with open('datasets/codeforces/train.jsonl', 'r') as f:
    train_data = [json.loads(line.strip()) for line in f if line.strip()]

with open('datasets/codeforces/test.jsonl', 'r') as f:
    test_data = [json.loads(line.strip()) for line in f if line.strip()]
 
train_prompts = []
test_prompts = []
   
for item in train_data:
    if not item['executable']:
        continue
    content = f'{system}\n\n'
    content += f'{item["description"]}\n\n'
    content += f'Input:\n\n{item["input_format"]}\n\n'
    if item["output_format"]:
        content += f'Output:\n\n{item["output_format"]}\n\n'
    if item["interaction_format"]:
        content += f'Interaction format\n\n{item["interaction_format"]}\n\n'
    if item["examples"]:
        for i, it in enumerate(item['examples']):
            content += f'Sample Input {i}\n\n{it["input"]}\n\n'
            content += f'Sample Output {i}\n\n{it["output"]}\n\n'
    content = content.strip()
    train_prompts.append({
            "contest_id": item["contest_id"], 
            "prompt": content,
            "official_tests": item["official_tests"],
            "generated_checker": item["generated_checker"] if item["generated_checker"] else None,
            "time_limit": item.get("time_limit", None),
            "memory_limit": item.get("memory_limit", None),
            "INPUT_MODE": item.get("input_mode", None),
        })

for item in test_data:
    if not item['executable']:
        continue
    content = f'{system}\n\n'
    content += f'{item["description"]}\n\n'
    if item["input_format"]:
        content += f'Input:\n\n{item["input_format"]}\n\n'
    if item["output_format"]:
        content += f'Output:\n\n{item["output_format"]}\n\n'
    if item["interaction_format"]:
        content += f'Interaction format\n\n{item["interaction_format"]}\n\n'
    if item["examples"]:
        for i, it in enumerate(item['examples']):
            content += f'Sample Input {i}\n\n{it["input"]}\n\n'
            content += f'Sample Output {i}\n\n{it["output"]}\n\n'
    content = content.strip()
    test_prompts.append({
            "contest_id": item["contest_id"], 
            "prompt": content,
            "official_tests": [item["official_tests"][0]] if item["official_tests"] else item["official_tests"],
            "generated_checker": item["generated_checker"] if item["generated_checker"] else None,
            "time_limit": item.get("time_limit", None),
            "memory_limit": item.get("memory_limit", None),
            "INPUT_MODE": item.get("input_mode", None),
        })
    
with open('datasets/codeforces/train_prompts.jsonl', 'w') as f:
    for item in train_prompts:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
with open('datasets/codeforces/test_prompts.jsonl', 'w') as f:
    for item in test_prompts:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')


