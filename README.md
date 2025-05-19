<div align="center">
<h1>CoT Compression</h1> 
</div>

# Introduction
This repository contains the code for CoT Compression.

# Repository Structure
```bash
cot_compression/
├── datasets
├── train_config
├── inference.py
├── evaluation.py
├── processing_data.py
├── get_lf_input.py
├── README.md
├── requirements.txt
```
# Installation
```bash
conda create -n cot_compression python=3.12

pip install -r requirements.txt
```

# Usage
## inference.py
Usage: `python inference.py [options]`

This script do inference on the model (both training and evaluation steps will use this script) and contains evaluation scripts.

### Options:
*   `--output_dir STR`: Directory to save predictions
*   `--model_path STR`: Path to the base language model
*   `--tokenizer_path STR`: Path to the tokenizer
*   `--adapter_path STR`: Path to the LoRA adapter
*   `--model_size STR`: Model size (e.g., `7b`)
*   `--use_adapter BOOL`: Whether to use the LoRA adapter
*   `--benchmark STR`: Benchmark to evaluate on (`r1`, `gsm8k`, `math`, `human_eval`). Note that `human_eval` will only do inference, evaluation will be done by `evaluate.py`.
*   `--data_type STR`: Data type to use (`train`, `test`)
*   `--max_num_examples INT`: Maximum number of examples to evaluate (random selection)
*   `--max_new_tokens INT`: Maximum number of new tokens to generate
*   `--temperature FLOAT`: Sampling temperature
*   `--seed INT`: Random seed
*   `--num_gpus INT`: Number of GPUs to use
*   `--num_per_example INT`: **This parameter is only needed when generating training data**, i.e. BoN parameter.
*   `--gpu_memory_utilization FLOAT`: GPU memory utilization
*   `--origin_path STR`: Path to the original data eval path
*   `--save_period INT`: Save predictions every N steps


## evaluate.py

Usage: `python evaluate.py [options]`

This script processes the predictions from a benchmark (currently, only `human_eval` benchmark needs to be processed).

### Options:

*   `--benchmark STR`: The benchmark to process
*   `--input_dir STR`: The input directory
*   `--pred_file STR`: The prediction file name

## processing_data.py
Usage: `python processing_data.py [options]`
This script processes the data for the benchmark, including filtering the data by correctness and CoT length.

### Options:
*   `--max_length INT`: Maximum filter length of CoT
*   `--input_dir STR`: The input directory
*   `--pred_file STR`: The prediction file name

PS: The output directory will be the same as the input directory, and the output file will be `predictions_filtered.jsonl`.

## get_lf_input.py
Usage: `python get_lf_input.py [options]`
This script generates the SFT data for LLaMAFactory

### Options:
*   `--data_name STR`: Output dataset name
*   `--llamafactory_dir STR`: Path to the LLaMA Factory data directory
*   `--input_dir STR`: Directory containing `predictions_filtered.jsonl`


# Codeforces Datasets

We also successfully fine-tuned and evaluated our models on the `open-r1/codeforces` dataset. 

The entire evaluation framework is built on Piston. We've pre-built Docker images which are available from Docker Hub.

The detailed procedure is outlined below.

## Load dataset

Download `codeforces` datasets, and put it in `datasets` directory

```
datasets/
├── codeforces/
	├── test_prmopts.jsonl
	├── test.jsonl
	├── train_prompts.jsonl
	├── test.jsonl
```

## Load docker image

Download `piston_codeforces.7z`, unzip to `.tar` and run the following command

```bash
docker load -i piston_codeforces.tar
```

or you can directly pull from docker hub

```bash
docker pull 1814071380/piston_codeforces:latest
```

## Run Docker Container

```bash
docker run -d \
  --name piston_codeforces \
  -e PORT=2000 \
  -e PISTON_COMPILE_TIMEOUT=60000 \
  -e PISTON_RUN_TIMEOUT=60000 \
  -e PISTON_OUTPUT_MAX_SIZE=1000000000 \
  -e PISTON_MAX_FILE_SIZE=1000000000 \
  -e PISTON_DISABLE_NETWORKING=true \
  -e PISTON_REPO_URL=https://ghfast.top/https://github.com/guipenedo/piston/releases/download/pkgs/index \
  -p 2000:2000 \
  1814071380/piston_codeforces:latest
```

PS: 

- `PISTON_REPO_URL` is proxy url, if you can directly access github repo, please change to `https://github.com/guipenedo/piston/releases/download/pkgs/index`
- If you change the port `[2000]`, remember to change the code in `evaluation.py`

## Inference & Evaluation

```bash
python inference.py --output_dir "output_dir" \
    --model_path "model_path" \
	--data_type "test" \
    --max_new_tokens 16384 \
	--gpu_memory_utilization 0.9 \
    --temperature 0.6 --benchmark "codeforces" \
	--num_gpus 2
```

```bash
python evaluation.py --input_dir "directory_that_contains_predictions.jsonl" --benchmark codeforces
```

