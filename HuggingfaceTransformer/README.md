# Summary

This environment is intended to use Huggingface Transformers python package through PythonCall to initialise SymbolicTransformer and compare results of calculations.


# Dependencies

Julia dependencies are CondaPkg,PythonCall and the parent SymbolicTransformer package.

Python dependencies are added using PythonCall:

Transformers based on [Transformers Installation](https://huggingface.co/docs/transformers/en/installation)

```
conda install conda-forge::transformers
```

Pytorch with cuda 12.1 based on [Pytorch Installation](https://pytorch.org/get-started/locally/)

```
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

Models are downloaded and cached in models folder


# Usage

To run this, start in the repos root folder

In Julia REPL, use `]` to switch to package manager.

```
activate HuggingfaceTransformer
```

Backspace to return to the Julia REPL prompt

```
using HuggingfaceTransformer
use_pythia_70m()
```

# Notes

The install of is several gb, mainly from pytorch. Transformer models can be much larger. The model referenced by use_pythia_70m is 161mb.
