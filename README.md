# SymbolicTransformer

This project works through some ideas around language model interpretability through Julia.

# Motivation

Language models consist of billions of numbers which are combined in a complicated pattern with other blocks of numbers which represent meaningful text to generate more numbers representing more text. The goal is to name symbols and operations representing different stages of this calculation so they can be reasoned about further

## Short Term Goal

To generate values representing inputs and outputs to a transformer language model which can be combined to perform the model's processing, and allow terms to be
expanded to show intermediate steps.

```julia

julia> using Transformers.HuggingFace

julia> using SymbolicTransformer

julia> encoder, model = hgf"EleutherAI/pythia-70m-deduped"


julia> T = prompt(model, encoder, "The capital of Ireland")
PromptedTransformer

julia> embed(T, " is")
Residual(" is")

julia> T * r
Residual(T * " is")

julia> :(T * r)
:(T * r)

julia> expand(:(T * r))
:(L4 * (L3 * (L2 * (L1 * r))))


```

## Medium Term Goal

To use attribution, gradients and estimation to identify and extract features relevant to a particular calculation through the model, and to neglect terms
which have lower relevance to a particular calculation.

```julia

julia> L1 * r
Key1 + V2 + r 

```

# Plans and Progress

I'm aiming to see the flow through using Transformers.jl with Pythia/GPTNeo-X models. Later it should be possible to abstract out the logic which doesn't directly depend on a specific implementation.


[![Build Status](https://github.com/prior-technology/SymbolicTransformer/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/prior-technology/SymbolicTransformer/actions/workflows/CI.yml?query=branch%3Amain)
