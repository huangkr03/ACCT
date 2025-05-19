import json


class Concode:
    DATASET_PATH = "datasets/code_x_glue_tc_text_to_code"

    def __init__(self):
        self.val = f'{self.DATASET_PATH}/dev.jsonl'
        self.train = f'{self.DATASET_PATH}/train.jsonl'
        with open("datasets/few_shot_examples/concode_few_shot_prompts.json", "r") as file:
            self.examples = json.load(file)
        self.system_prompt = "Answer the following instructions in a one line of Java code:\n"
        
    def get_prompt_two_shot(self, line):
        text = line["nl"].split("concode_field_sep")[0].strip()
        if text.endswith("."):
            text = text[:-1].strip()
        prompt = f"e.g.\nInstruction:\n{self.examples['instruction1']}\nAnswer:\n```java\n{self.examples['solution1']}\n```\
                   \nInstruction:\n{self.examples['instruction2']}\nAnswer:\n```java\n{self.examples['solution2']}\n```\
                   \nNow please answer the follow instruction:\n{text}"
        return self.system_prompt + prompt
    
    def get_dataset(self, train=False):
        # test split of the dataset doesn't have targets
        if train:
            with open(self.train, "r") as file:
                lines = file.readlines()
        else:
            with open(self.val, "r") as file:
                lines = file.readlines()
        dataset = []
        for i, line in enumerate(lines):
            line = json.loads(line)
            dataset.append({
                "id": i,
                "nl": line["nl"],
                "prompt": self.get_prompt_two_shot(line),
                "answer": line["code"],
            })
        return dataset
        
        
class Conala:
    DATASET_PATH = "datasets/conala"

    def __init__(self):
        self.val = f'{self.DATASET_PATH}/test.jsonl'
        self.train = f'{self.DATASET_PATH}/train.jsonl'
        with open("datasets/few_shot_examples/conala_few_shot_prompts.json", "r") as file:
            self.examples = json.load(file)
        self.system_prompt = "Answer the following instructions in a one line of Python code:\n"
        
    def get_prompt_two_shot(self, line):
        text = line["rewritten_intent"]
        prompt = f"e.g.\nInstruction:\n{self.examples['instruction1']}\nAnswer:\n```python\n{self.examples['solution1']}\n```\
                   \nInstruction:\n{self.examples['instruction2']}\nAnswer:\n```python\n{self.examples['solution2']}\n```\
                   \nNow please answer the follow instruction:\n{text}"
        return self.system_prompt + prompt

    def get_dataset(self, train=False):
        # test split of the dataset doesn't have targets
        if train:
            with open(self.train, "r") as file:
                lines = file.readlines()
        else:
            with open(self.val, "r") as file:
                lines = file.readlines()
        dataset = []
        for i, line in enumerate(lines):
            line = json.loads(line)
            dataset.append({
                "id": i,
                "nl": line["rewritten_intent"],
                "prompt": self.get_prompt_two_shot(line),
                "answer": line["snippet"],
            })
        return dataset