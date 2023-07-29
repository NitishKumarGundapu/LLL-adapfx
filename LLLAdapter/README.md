# Lifelong Language Learning With Adapter Based Transformers
Continual Learning is important for real-world natural language processing applications, where computational systems are required to interact with continuous streams of  tasks and language over time.  When forced to adapt to new tasks and inputs, language models experience catastrophic forgetting. The current generative replay-based algorithms are not scalable to many tasks, and their performance may degrade from a change in the task order. In this paper, we propose a model based on network growth - a pre-trained Transformer with Adapter modules for each task - that sequentially learns new NLP tasks in various domains and prevents catastrophic forgetting without retraining the model from scratch. We train and maintain light weight  adapter modules sequentially for each task. Without increasing network growth by more than 15\% and avoiding replay and task order bias, the current design allows us to increase average task accuracy by 1.3\% over the  baseline models.

## Dataset

| Task | Dataset (Original Data Link) |
| ---- | ------- |
| Sentiment Analysis  | [SST](https://nlp.stanford.edu/sentiment/treebank.html) |
| Semantic Role Labeling | [QA‑SRL](https://dada.cs.washington.edu/qasrl/) |
| Goal-Oriented Dialogue | [WOZ](https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz) |

In order to unify the format of all the dataset, we first ran the code in https://github.com/salesforce/decaNLP to get the tranformed dataset, and then converted them into Squad-like format. For the last 5 dataset, we converted them directly. All converted dataset are available [here](https://drive.google.com/file/d/1rWcgnVcNpwxmBI3c5ovNx-E8XKOEL77S/view?usp=sharing).

## Dependencies
- Ubuntu >= 16.04
- This code only supports the following GPUs:
  - NVIDIA Geforce RTX 2080TI 
  - NVIDIA TESLA V100
- python3
- cuda 10.1
- python packages are listed in `requirements.txt`

## Setup
1. Create the following two directories in wherever you want. (you can name the directories arbitrarily):
    - `data directory`: Where the dataset will be load by the model.
    - `model directory`: The place for the model to dump its outputs.
2. Download the dataset: Download [here](https://drive.google.com/file/d/1rWcgnVcNpwxmBI3c5ovNx-E8XKOEL77S/view?usp=sharing) and decompress it. After decompression, move all the files in the decompressed directory into `data directory`.
3. In `env`, set the value of DATA_DIR as `data directory` and set the value of  MODEL_ROOT_DIR as `model directory`.
3. Install packages `pip install -r requirements.txt`, It will install all the required pacakges.

## Training and Testing

`train.py` and `test.py` are the entrance for training and testing. Main options for them include:

| Options        | Description   |
| -------------  | ------------- |
| seq_train_type | The mode to deal with a sequence of tasks. Mode include: lll\|finetune\|multitask\|mas\|ewc\|gem. "lll" is the default value corresponding our proposed method. The others are the methods for comparing with our proposal. |
| tasks          | A sequence of tasks we want to train by seq_train_type. Leave a space between tasks after the `--tasks` tag. Tasks are the keys in TASK_DICT variable in `settings.py` |
| model_name     | The language model we want to use. The default is `gpt2`. Options include gpt2\|openai-gpt, |
| gen_lm_sample_percentage | This tag only works with `--seq_train_type lll`. The percentage of the size of the dataset will be generated as pseudo samples for our proposed method. |
| lm_lambda      | Lambda value for the loss function. |
| max_n_epochs   | Maximum epoch value for all tasks. |
| min_batch_size | Minimum batch size for all tasks. |
| min_n_steps    | Minimum step for optimizing the model for all tasks. |
| n_train_epochs | Epochs for training for all tasks. |
| n_gpu          | Number of gpu to be used. |
| reg_lambda     | Lambda value for mas and ewc. |
| top_k_lm       | Top k sampling for the language model. |
| top_k_qa       | Top k sampling for the qa model. |
| train_batch_size | Batch size for all tasks. The default is 0. Once the value equals to 0, The batch size will be decided dynamically based on the memory usage of the gpu. |

### Training 

#### Example:

If you want to train sst, srl and woz.en sequentially by our proposed method, run:
```bash
 --seq_train_type lll --model_name gpt2 --n_gpus 8 --n_workers 75 --fp32 --n_train_epochs 10  --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0
```

#### Outputs:


If you run:
```bash
python train.py --data_dir $DATA_DIR --model_dir_root $MODEL_ROOT_DIR --seq_train_type lll --model_name gpt2 --n_gpus 8 --n_workers 75 --fp32 --n_train_epochs 12  --gen_lm_sample_percentage 0.00 --tasks sst srl woz.en --lm_lambda 0.0
```
Then the models will be dumped in the following directories: `$MODEL_ROOT_DIR/gpt2/lll/sst_srl_woz.en_0.2/sst`, `$MODEL_ROOT_DIR/gpt2/lll/sst_srl_woz.en_0.2/srl`, `$MODEL_ROOT_DIR/gpt2/lll/sst_srl_woz.en_0.2/woz.en`.


### Testing

#### Example:

This example test the model trained on sst, srl and woz.en by lll method.
```bash
python train.py --data_dir $DATA_DIR --model_dir_root $MODEL_ROOT_DIR --seq_train_type lll --n_workers 75 --fp32 --gen_lm_sample_percentage 0.00 --n_train_epochs 12 --lm_lambda 0.0 --tasks sst srl woz.en
```

#### Outputs:
After running testing program, the metrics: `metrics.json` will be dumped in the same directory of Training's outputs.

