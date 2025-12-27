# BERT MULTI-GPU PRE-TRAIN ON ONE MACHINE WITHOUT HOROVOD (Data Parallelism)

[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

# REASONABLE / PRINCIPLE

More gpu means more data in a batch (, batch size is larger). And the gradients of a batch data is averaged for back-propagation.

If the sum learning rate of one batch is fixed, then the learning rate of one data is smaller, when batch size is larger.

If the learning rate of one data is fixed, then the sum learning rate of one batch is larger, when batch size is larger.

**It seems:** More gpu --> Larger sum learning rate of one batch --> Faster training. 

But as the learning rate can only be in the range of 1e-4 to 1e-5, [**train_samples_per_second**](https://github.com/guotong1988/LLM-post-training/issues/1) becomes the evaluation metric for speed. The total batch size is the determining factor. Training will fail if the learning rate is set too high.

# WHATS NEW

Using 1-GPU (100 batch size) vs using 4-GPU (400 batch size) for the same learning rate (0.00001) and same pre-training steps (1,000,000) will be no difference of 0.1% in downstream task accuracy.

# REQUIREMENT

python 3

tensorflow 1.14 - 1.15

# TRAINING

0, edit the input and output file name in `create_pretraining_data.py` and `run_pretraining_gpu.py`

1, run `create_pretraining_data.py`

2, run `run_pretraining_gpu_v2.py`

# PARAMETERS

Edit `n_gpus` in `run_pretraining_gpu_v2.py`


# DATA

In `sample_text.txt`, sentence is end by `\n`, paragraph is splitted by empty line.

# EXPERIMENT RESULT ON DOWNSTREAM TASKS

Quora question pairs English dataset,

Official BERT: ACC 91.2, AUC 96.9

This BERT with pretrain loss 2.05: ACC 90.1, AUC 96.3

# NOTE

### 1)
For `HierarchicalCopyAllReduce` `MirroredStrategy`, `global_step/sec` shows the sum of multi gpus' steps.
### 2)
`batch_size` is the `batch_size` per GPU, not the `global_batch_size`


