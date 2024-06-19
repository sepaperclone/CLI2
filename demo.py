import random
import numpy as np
import torch
import os
import pandas as pd
import json

import argparse


from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm import  SamplingParams


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

code1 = """
public static void main (String [] args) throws InterruptedException {
    AtomicInteger counter = new AtomicInteger ();
    ExecutorService pool = Executors.newFixedThreadPool (4);
    for (int i = 0;
    i < 4; ++ i) {
        pool.execute (new MyTask (counter));
    }
    pool.shutdown ();
    pool.awaitTermination (1, TimeUnit.HOURS);
}
"""
code2 = """
#include <stdio.h> 
#include <stdlib.h> 
#include <pthread.h> 
#include <time.h>  

int atm_counter = 0;
pthread_mutex_t counter_lock;

void *mytask(void *arg) 
{
    pthread_mutex_lock(&counter_lock); 
    atm_counter++;
    pthread_mutex_unlock(&counter_lock); 
    return NULL; 
}

int main()
{
    pthread_t threads[4];

    for (int i = 0; i < 4; i++)
        pthread_create(&threads[i], NULL, mytask, NULL);
  
    for (int i = 0; i < 4; i++)
        pthread_join(threads[i], NULL);
 
    sleep(3600);

    return 0;
}

"""

query_temp = "You will be given a pair of code snippets"\
                    "you need to analyze the functionality of them, then determine if these two code snippets are code clone"\
                    "and explain why according to the data of code snippet 1 and code snippet 2."\
                    "respond with yes if they are clone, otherwise respond with no."\
                    "Code clone means two code snippets share the same functionality or they are different implementations of the same problem."\
                    "Each code snippet is delimited by triple backticks."\
                    "\n[Code Snippet 1]: ```\n{code1}```\n[Code Snippet 2]: ```\n{code2}```"
       
messages = [
        {"role": "system", 
        "content": "You are a helpful code assistant."\
            " You are very familiar with analyzing code logic and summarizing it."\
            " You can compare the functionality of two code segments and determine whether they are code clones."
        },
        {"role": "user", "content": query_temp.format(code1=code1, code2=code2)}
    ]
test_queries = [messages]
rs = chat(test_queries)

print("=== code pairs for detecting ===")
print("=== === code1 === ===")
print(code1)
print("=== === code2 === ===")
print(code2)
print("=== === response === ===")
print(rs)
