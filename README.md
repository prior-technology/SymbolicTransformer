# SymbolicTransformer

This project provides symbolic notation and manipulation tools for transformer language model interpretability in Julia, designed to work with [TransformerAlgebra](../TransformerAlgebra) (Python).

## Architecture

SymbolicTransformer delegates model operations to Python via [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl), while providing Julia-native symbolic expression types and manipulation.

- **Interface.jl** - Reference types and symbolic expression types (Embedding, Unembedding, Residual, BlockContrib, etc.)
- **PythonBridge.jl** - Connection to Python's TransformerAlgebra service
- **LayerNormalization.jl** - Pure Julia layer normalization implementation

## Installation

```julia
using Pkg
Pkg.develop(path="path/to/SymbolicTransformer")
```

## Dependencies

- `JSON3` - JSON serialization for Python communication
- `PythonCall` - Julia-Python interop
- `LaTeXStrings` - LaTeX formatting support
- `LinearAlgebra` - Standard library

## Usage

```julia
using SymbolicTransformer

# Connect to Python's TransformerAlgebra
bridge = connect("EleutherAI/pythia-160m-deduped")

# Analyze a prompt
analysis = analyze(bridge, "The capital of Ireland")

# Get top predictions
preds = top_predictions(analysis)

# Decompose a logit into per-block contributions
decompose(analysis, " Dublin")
```

## Symbolic Types

The package provides symbolic reference and expression types:

### References (point to data in Python)
- `TokenRef` - Reference to a token in vocabulary
- `EmbeddingRef` - Reference to embedding vector
- `UnembeddingRef` - Reference to unembedding vector
- `ResidualRef` - Reference to cached residual vector
- `BlockContribRef` - Reference to block contribution

### Expressions (for symbolic manipulation)
- `Embedding`, `Unembedding` - Vector expressions
- `Residual`, `BlockContrib` - Residual stream expressions
- `Sum`, `Scaled` - Composite expressions
- `LayerNorm` - Layer-normalized expression
- `InnerProduct` - Scalar expression (logit)

## Expansion Rules

Expressions can be expanded to show intermediate computations:

```julia
# Expand residual into embedding + block contributions
# x^L_j = embed_j + Dx^1_j + Dx^2_j + ... + Dx^L_j
expanded = expand(residual, n_layers)

# Expand inner product through layer norm
# <y, LN(a + b)> = scale * (<g*y, c(a)> + <g*y, c(b)>) + <y, beta>
expanded = expand(inner_product)
```

## Status

This package is under active development. The Julia SymbolicTransformer and Python TransformerAlgebra are designed to work together - Julia handles symbolic manipulation while Python handles model operations via HuggingFace transformers.

[![Build Status](https://github.com/prior-technology/SymbolicTransformer/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/prior-technology/SymbolicTransformer/actions/workflows/CI.yml?query=branch%3Amain)
