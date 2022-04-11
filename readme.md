# Simple Annotated implementation of GPT-NeoX in PyTorch

This is a simpler implementation of [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) in PyTorch.
We have taken out several optimizations in GPT-NeoX for simplicity.

### [Annotated implementation](https://lit.labml.ai/github/labmlai/neox/tree/main/src/neox/__init__.py)

### Sample usages

* [Notebook to download checkpoints and test the model](https://github.com/labmlai/neox/tree/main/notebooks/download_and_evaluate.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai//neox/tree/main/notebooks/download_and_evaluate.ipynb)
* [Fine-tuning example](https://lit.labml.ai/github/labmlai/neox/tree/main/src/neox/samples/fine_tune_biases.py)
* [Text generating on multi GPU](https://lit.labml.ai/github/labmlai/neox/tree/main/src/neox/samples/generating_pipe.html)
* 🚧 [Text generating on a single GPU](https://lit.labml.ai/github/labmlai/neox/tree/main/src/neox/samples/generating_single_gpu.html)
