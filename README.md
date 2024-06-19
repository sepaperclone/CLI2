# This is the Repo for ASE 2024 CLI2 clone detector

## Repo structure
- dataset: the folder contains all the clone dataset, summmary, you need to download from the huggingface(https://huggingface.co/datasets/AnomalyPaper/CLI2_Dataset) because the file size limit.
- models: the fine-tuned model weights, download from huggingface (https://huggingface.co/AnomalyPaper/CLI2).
- README.md: readme file

## Dependency and Data Preparing:
our experiment enviroment dependend on pytorch, cuda, and vllm. Conda is reconmended for managing the enviroments. If you want to run the model we released, you need to install the following packages:
```bash
conda create -n vllm python=3.9  # create a conda enviroment
# pytorch
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install vllm=0.4.2
pip install transformers=4.40.1
```
you can also use the enviroment we build, which may take longer time:
```bash
conda env create -f vllm.yml 
```
### Downloading Model
CLI2 is based on Qwen-1.8B, you can download the model from huggingface([https://huggingface.co/Qwen/Qwen1.5-1.8B](https://huggingface.co/AnomalyPaper/CLI2))

### Llama Factory Dependency
If you want to reproduce our experiment including the fine-tuning step, you need to download the Llama-Factory (https://github.com/hiyouga/LLaMA-Factory/), or use other fine-tuning framework you prefered.
the dataset is organized as **alpaca** format.
there are 3 .json files the dir *dataset*:
- fine_tuning_summary.json: dataset for the first stage fine-tuning
- fine_tuning_mixed.json: dataset for the second stage fine-tuning
- fine_tuning_dpo.json: dataset for the third stage fine-tuning

Use LLama-Factory as example, you need to copy all three files into dir `LLaMa-Factory/data/`, then add the following data to 
`LLaMa-Factory/data/dataset_info.json`
```json
"clone_summary": {
    "file_name": "fine_tuning_summary.json",
    "columns": {
      "prompt": "instruction",
      "response": "output",
      "system":"system",
      "history": "history"
    }
  },
"clone_mixed": {
    "file_name": "fine_tuning_mixed.json",
    "columns": {
      "prompt": "instruction",
      "response": "output",
      "system":"system",
      "history": "history"
    }
  },
  "clone_dpo": {
    "file_name": "fine_tuning_dpo.json",
    "ranking":true,
    "columns": {
      "prompt": "instruction",
      "response": "output",
      "system":"system",
      "history": "history"
    }
  },
```
then you can use the script provided in `LLaMa-Factory/examples/lora` to run the fine-tuning experiment with the specific dataset.

## Usage

### Downloading Model
CLI2 is published on huggingface: 

### Run a Demo
We provide a python script to load the model and detect a pair of code snippets. you can change the two codes into any code snippet you prefered.
```bash
conda activate vllm
python demo.py --base_path qwen1_8b # provide the model path you downloaded.
```
### Run the Test Experiment:
You can use `eval.py` to run the test experiment of our paper.
```bash
python eval.py --base_model qwen1_8b
```
