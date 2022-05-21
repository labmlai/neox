# Simple Annotated implementation of GPT-NeoX in PyTorch

This is a simpler implementation of [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) in PyTorch. We have taken out
several optimizations in GPT-NeoX for simplicity.

### [Annotated implementation](https://lit.labml.ai/github/labmlai/neox/tree/main/src/neox/__init__.py)

[![Screenshot of annotated implementation](https://github.com/labmlai/neox/blob/main/assets/annotated_gpt_neox_model.png)](https://lit.labml.ai/github/labmlai/neox/tree/main/src/neox/__init__.py)

### Sample usages

* [Notebook to download checkpoints and test the model](https://github.com/labmlai/neox/blob/main/notebooks/download_and_evaluate.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai//neox/blob/main/notebooks/download_and_evaluate.ipynb)
* [Fine-tuning example](https://lit.labml.ai/github/labmlai/neox/tree/main/src/neox/samples/fine_tune_biases.py)
* [Text generating on multi GPU](https://lit.labml.ai/github/labmlai/neox/tree/main/src/neox/samples/generating_pipe.html)
* [Text generating on a small GPU (offloading to CPU)](https://lit.labml.ai/github/labmlai/neox/tree/main/src/neox/samples/generating_small_gpu.html)
* [Text generating on a single 48GB GPU](https://lit.labml.ai/github/labmlai/neox/tree/main/src/neox/samples/generating_gpu.html)

### [Playground](https://neox.labml.ai)

[![Screenshot of playground](https://github.com/labmlai/neox/blob/main/assets/gpt_neox_playground.png)](https://neox.labml.ai)


### Evaluation

| Task       | Metric          | NeoX Impl (2 GPU) | This repo (1 GPU) |
|------------|-----------------|-------------------|-------------------|
| anli_r1    | acc             | 0.3270            | 0.3360            |
|            | acc_stderr      | 0.0148            | 0.0149            | 
| anli_r2    | acc             | 0.3410            | 0.3350            |
|            | acc_stderr      | 0.0150            | 0.0149            |
| anli_r3    | acc             | 0.3567            | 0.3525            |
|            | acc_stderr      | 0.0138            | 0.0149            |
| hellaswag  | acc             | 0.5351            | 0.5353            |
|            | acc_stderr      | 0.0050            | 0.0050            |
|            | acc_norm        | 0.7140            | 0.7145            |
|            | acc_norm_stderr | 0.0045            | 0.0045            |
| lambada    | acc             | 0.7211            | 0.7204            |
|            | acc_stderr      | 0.0062            | 0.0063            |
|            | ppl             | 3.6760            | 3.6375            |
|            | ppl_stderr      | 0.0760            | 0.0747            |
| piqa       | acc             | 0.7748            | 0.7758            |
|            | acc_stderr      | 0.0097            | 0.0097            |
|            | acc_norm        | 0.7786            | 0.7845            |
|            | acc_norm_stderr | 0.0097            | 0.0096            |
| winogrande | acc             | 0.6598            | 0.6582            |
|            | acc_stderr      | 0.0133            | 0.0133            |
| wsc        | acc             | 0.5096            | 0.5000            |
|            | acc_stderr      | 0.0493            | 0.0493            |
