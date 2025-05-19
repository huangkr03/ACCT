

import json
import re


def detect_infinit_loops(text, loop_length=200, loop_threshold=4):
    loop_content1 = text[-loop_length - 1 : -1]
    loop_content2 = text[- 2 * loop_length - 1 : -loop_length - 1]
    # if loop_content appears more than 3 times in the text, it is an infinite loop
    if text.count(loop_content1) >= loop_threshold or text.count(loop_content2) >= loop_threshold:
        return True
    return False
    

def extract_code_block(text, language='python'):
    # regex = r'```python\n(.*?)\n```'
    regex = r'```' + language + r'\n(.*?)\n```'
    code_blocks = re.findall(regex, text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1]
    return ''
    
# infinit = 0

# total = 0
# error = 0

# with open("/home/keruihuang/cot_compression/outputs/DeepSeek-R1-Distill-Qwen-7B/baseline-16k/taco/7b/train/samples/predictions.jsonl", "r") as f:
#     for line in f:
#         data = json.loads(line)
#         print(data.keys())
#         if detect_infinit_loops(data["model_output"]):
#             infinit += 1
#             if data['finish_reason'] == "stop":
#                 error += 1
#                 # with open("temp.txt", "w") as f:
#                     # f.write(data["model_output"])
#                 # print(data["model_output"])
#                 # break
#         total += 1
    
# print("infinit loops: ", infinit)
# print("total: ", total)
# print("error: ", error)