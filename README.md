# This is the Repo for CLI2 clone detector
## Additional Experiments 
Due to the pape limitation of the paper, we will public part of the experiment results here.
### Dataset Statistic

| Statistic | GoogleCodeJam2 (Lines) | CodeNet (Dou's) (Lines) | GoogleCodeJam2 (Tokens) | CodeNet (Dou's) (Tokens) |
|-----------|------------------------|-------------------------|-------------------------|--------------------------|
| Max       | 942                    | 1052                    | 53198                   | 7900                     |
| Min       | 10                     | 2                       | 31                      | 30                       |
| Mean      | 75.08                  | 43.82                   | 539.99                  | 286.06                   |
| Median    | 59.0                   | 34.0                    | 399.0                   | 212.0                    |

GCJ2 exhibits higher complexity compared to CodeNet (Dou's), as evidenced by the following statistics:

- Lines: GCJ2 has a maximum of 942 lines, a minimum of 10 lines, a mean of 75.08 lines, and a median of 59.0 lines. In contrast, CodeNet (Dou's) has a maximum of 1052 lines, a minimum of 2 lines, a mean of 43.82 lines, and a median of 34.0 lines.

- Tokens: GCJ2 has a maximum of 53198 tokens, a minimum of 31 tokens, a mean of 539.99 tokens, and a median of 399.0 tokens. CodeNet (Dou's) has a maximum of 7900 tokens, a minimum of 30 tokens, a mean of 286.06 tokens, and a median of 212.0 tokens.

### Performance of LLM under One-Shot

We evaluated the One-Shot performance of several open-source models. However, during the testing, we found that although the performance improved, the proportion of refusals to answer increased significantly. Most long and complex code segments were refused, making it difficult to judge whether One-Shot truly improved the model's performance. In contrast, Deepseek_7B did not show a significant increase in refusals, and its performance did not improve at all.

#### One-Shot

| Model        | Precision (P) | Recall (R) | F1 Score (F1) |  Fail   |
|--------------|---------------|------------|---------------|---------|
| codellama-34b| 0.281         | 0.941      | 0.432         |**76%**  |
| deepseek_33b | 0.567         | 0.81       | 0.667         |**71.4%**|
| deepseek_7b  | 0.188         | 0.894      | 0.311         |12.3%    |

#### Zero-Shot

| Model        | Precision (P) | Recall (R) | F1 Score (F1) |  Fail |
|--------------|---------------|------------|---------------|-------|
| codellama-34b| 0.172         | 0.595      | 0.267         |19.67% |
| deepseek_33b | 0.226         | 0.98       | 0.368         |2%     |
| deepseek_7b  | 0.2           | 0.711      | 0.312         |10%    |

### Importance of Multi-Stage Fine-tuning

To validate the effectiveness of each stage in our multi-stage fine-tuning process, we conducted experiments for each fine-tuning stage, continuing training with the same number of steps to observe whether the model's performance could still reach an optimal level without adopting new fine-tuning methods. For example, the F1 score in the first row of the table uses only the first stage of fine-tuning without switching fine-tuning methods with the same number of training steps.
We can found that because the validation set is used to determine the optimal model checkpoint, there are instances where further training yields better results on test dataset. However, these improvements do not surpass the performance achieved in the subsequent stage of fine-tuning.

| F1 Score          | Step-1 | Step-2 | Step-3 |
|-------------------|--------|--------|--------|
| Stage-1           | 0.640  | *0.664*  | 0.622  |
| Stage-1 + Stage-2 | 0.640  | 0.715  | 0.713  |
| CLI2 (All 3 Stages) | 0.640  | **0.715**  | **0.760**  |

As shown in the table above, each stage has a performance ceiling, and therefore, it is not possible to achieve the effects of multi-stage fine-tuning by simply increasing the number of fine-tuning steps.

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
