# This is the Repo for CLI2 clone detector

## Repo structure
- dataset: the folder contains all the clone dataset, summmary, you can download them from the huggingface[(https://huggingface.co/datasets/AnomalyPaper/CLI2_Dataset)].
  - gcj4.pkl: A Dataframe of pandas which contains all the code snippets and their functionality category.
  - test_cross_fun.npy: A numpy array of test clone pairs from different functionality, which organized as [id1, id2, 0|1]
  - test_cross_lan.npy: Numpy array of clone pairs from different language from fine-tuning dataset.
- demo.py: script for running the detecting demo.
- eval.py: script for running the eval experiment.
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
CLI2 is based on Qwen-1.8B, you can download the model from huggingface([https://huggingface.co/AnomalyPaper/CLI2])

### Llama Factory Dependency
**If you want to reproduce our fine-tuning experiment**, you need to download the Llama-Factory (https://github.com/hiyouga/LLaMA-Factory/), or use other fine-tuning framework you prefered.
the dataset is organized as **alpaca** format.
there are 3 `.json` files the dir *dataset*(or you need to download from huggingface[((https://huggingface.co/datasets/AnomalyPaper/CLI2_Dataset))]):
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
CLI2 is published on huggingface: https://huggingface.co/AnomalyPaper/CLI2

### Run a Demo
We provide a python script to load the model and detect a pair of code snippets. you can change the two codes into any code snippet you prefered.
```bash
conda activate vllm
python demo.py --base_path qwen1_8b # provide the model path you downloaded.
```

#### example
*The final prompt in demo.py is as following*:

You will be given a pair of code snippetsyou need to analyze the functionality of them, then determine if these two code snippets are code cloneand explain why according to the data of code snippet 1 and code snippet 2.respond with yes if they are clone, otherwise respond with no.Code clone means two code snippets share the same functionality or they are different implementations of the same problem.Each code snippet is delimited by triple backticks.
[Code Snippet 1]: 
```java
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
```
[Code Snippet 2]: 
```cpp
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
```

*The output*:

[Functionality of Code Snippet 1]:This Java code is used to manage a pool of tasks that need to be executed concurrently. It uses an ExecutorService to create a pool of worker threads, each responsible for executing a single task. The tasks are incremented in a counter before being executed. The main thread waits for all the worker threads to finish before exiting.

[Functionality of Code Snippet 2]:This code is a solution to a problem from programing competition. It involves creating a thread for each test case and joining on the thread when it's done. The thread increments a counter every second and prints the current count at the end of the program. The main function creates four threads and joins on the threads when they're done. The threads are created using the `pthread_create` function.

[Answer]:According to their functionalities, they are different implementations of the same functionality, so the answer is yes, they are code clone.


### Run the Test Experiment:
You can use `eval.py` to run the test experiment of our paper.
```bash
python eval.py --base_model qwen1_8b
```
