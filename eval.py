import random
import numpy as np
import torch
import os
import pandas as pd
import json
import argparse

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from sklearn.metrics import precision_recall_fscore_support


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default=None, type=str, required=True)
args = parser.parse_args()
model_path = args.model_path

tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(model=model_path, tokenizer=model_path, max_model_len=8192)

sampling_params = SamplingParams(temperature=0.5,top_k=1, top_p=0.8, repetition_penalty=1.05, max_tokens=800)

def chat(messages_list):
    prompts = [tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) for messages in messages_list]
    # print(len(prompts))
    sampling_params.stop = [tokenizer.eos_token]
    outputs = llm.generate(prompts, sampling_params)
    generated_text = [output.outputs[0].text for output in outputs]
    return generated_text

def parse_output(output):
    if 'so the answer is no,' in output:
        return -1
    elif 'so the answer is yes,' in output:
        return 1
    else:
        return 0

def get_f1(rs, labels):
    preds = []
    for r in rs:
        if '[Answer]' not in r:
            preds.append(parse_output(r.lower()))
        else:
            index = r.find('[Answer]')
            r = r[index:]
            preds.append(parse_output(r.lower()))

    pred_data  = np.array([labels, preds]).T
    pred_data = pred_data[pred_data[:, 1]!=0]
    p, r, f1, _ = precision_recall_fscore_support(pred_data[:, 0], pred_data[:, 1], average='binary')
    print(f"Precision: {round(p, 3)}, Recall: {round(r, 3)}, F1: {round(f1, 3)}")
    return f1


import tiktoken
tik_tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

def get_test_queries(test_data, test_count, code_dict):
    np.random.shuffle(test_data)
    labels = [] 
    test_queries = []
    for d in test_data[:test_count]:
        query_temp = "You will be given a pair of code snippets"\
                    "you need to analyze the functionality of them, then determine if these two code snippets are code clone"\
                    "and explain why according to the data of code snippet 1 and code snippet 2."\
                    "respond with yes if they are clone, otherwise respond with no."\
                    "Code clone means two code snippets share the same functionality or they are different implementations of the same problem."\
                    "Each code snippet is delimited by triple backticks."\
                    "\n[Code Snippet 1]: ```\n{code1}```\n[Code Snippet 2]: ```\n{code2}```"
        if (d[0] not in code_dict.keys()) or (d[1] not in code_dict.keys()):
            continue
        code1, code2 = code_dict[d[0]], code_dict[d[1]]

        if (len(tik_tokenizer.encode(code1)) >= 1000) or (len(tik_tokenizer.encode(code2))>=1000):
            continue
        messages = [
                {"role": "system", 
                "content": "You are a helpful code assistant."\
                    " You are very familiar with analyzing code logic and summarizing it."\
                    " You can compare the functionality of two code segments and determine whether they are code clones."
                },
                {"role": "user", "content": query_temp.format(code1=code1, code2=code2)}
            ]
        test_queries.append(messages)
        labels.append(d[2])
    return labels, test_queries


def eval_F():
    print("=== Eval for Cross Functionality Performance ===")
    test_data = np.load('./dataset/test_cross_fun.npy')
    test_df = pd.read_pickle('./dataset/gcj/gcj4.pkl')
    fids = [int(i) for i in test_df['funid'].tolist()]
    test_df = test_df[test_df['funid'].isin(fids)]
    code_dict = dict(zip([i for i in test_df['funid'].tolist()], test_df['flines'].tolist()))
    test_count = 2000

    labels, test_queries = get_test_queries(test_data, test_count, code_dict)
    print("=== === Running Eval: ", model_path)
    f1s = []
    for i in range(10):
        labels, test_queries = get_test_queries(test_data)
        rs = chat(test_queries)
        f1s.append(get_f1(rs, labels))
    print(f1s, np.mean(f1s))

def eval_L():
    print("=== Eval for Cross Language Performance ===")
    test_data = np.load('./dataset/test_cross_lan.npy')
    test_df = pd.read_pickle('./dataset/gcj/gcj4.pkl')
    fids = [int(i) for i in test_df['funid'].tolist()]
    test_df = test_df[test_df['funid'].isin(fids)]
    code_dict = dict(zip([i for i in test_df['funid'].tolist()], test_df['flines'].tolist()))
    test_count = 2000

    labels, test_queries = get_test_queries(test_data, test_count, code_dict)
    print("=== === Running Eval: ", model_path)
    f1s = []
    for i in range(10):
        labels, test_queries = get_test_queries(test_data)
        rs = chat(test_queries)
        f1s.append(get_f1(rs, labels))
    print(f1s, np.mean(f1s))

eval_F()
eval_L()
