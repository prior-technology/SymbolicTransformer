# SymbolicTransformer

This project works through some ideas around language model interpretability through Julia.

# Motivation

Language models consist of billions of numbers which are combined in a complicated pattern with other blocks of numbers which represent meaningful text to generate more numbers representing more text. The goal here is to name symbols representing different stages of this calculation.

# End Goal

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


# Plans

I'm aiming to see the flow through using Transformers.jl with Pythia/GPTNeo-X models. Later it should be possible to abstract out the logic which doesn't directly depend on a specific implementation.

[![Build Status](https://github.com/prior-technology/SymbolicTransformer/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/prior-technology/SymbolicTransformer/actions/workflows/CI.yml?query=branch%3Amain)
