# SymbolicTransformer

This project works through some ideas around language model interpretability through Julia.

# Motivation

Language models consist of billions of numbers which are combined in a complicated pattern with other blocks of numbers which represent meaningful text to generate more numbers representing more text. The goal here is to name symbols representing different stages of this calculation.

# End Goal

```julia

julia> using Transformers.HuggingFace

julia> using SymbolicTransformer

julia> encoder, model = hgf"EleutherAI/pythia-70m-deduped"


julia> T = prompt(model, "The capital of Ireland")
PromptedTransformer

julia> embed(" is")
Residual(" is")

julia> T * r
Residual(T * " is")

julia> :(T * r)
:(T * r)

julia> expand(:(T * r))
:(L4 * (L3 * (L2 * (L1 * r))))

```

# Get Started

In the repo folder press `]` to enter REPL mode then use `activate .` to load the package.



[![Build Status](https://github.com/prior-technology/SymbolicTransformer/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/prior-technology/SymbolicTransformer/actions/workflows/CI.yml?query=branch%3Amain)
