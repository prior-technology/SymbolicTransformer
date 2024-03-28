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

I'm aiming to see the flow through using Transformers.jl with Pythia/GPTNeo-X models. Later it should be possible to abstract out the logic which doesn't directly depend on a specific implementation. Earlier work started to rewrite the algorithm from scratch, and earlier again focussed on abstract operations without specific implementations.

`WrappedTransformer` represents the results of calculations in types like `HGFResidual`. These include an expression which tracks the origin of the associated result. 

`PromptedTransformer` represents a specific transformer algorithm with prompt text. This acts on a residual vector using the `*` operation to run the internal blocks, returning the residual vector in the last position of the output layer (i.e. excluding input and output embedding layers). 

`predict` is a function which runs the model and calculates logits and probabilities for all tokens, returning each as a HGFResidual which includes an expression which should perform a similar calculation (returning only logits since probabilities depends on all logits for other tokens)

```julia
julia> T = prompt(model, encoder, "1, 2, 3, 4")
PromptedTransformer(Transformers.HuggingFace.HGFGPTNeoXModel, GPT2TextEncoder, "1, 2, 3, 4")

julia> r = first(embed(T, ","))
HGFResidual(",", embed(","))

julia> y = T * r
HGFResidual("1, 2, 3, 4,", T * embed(","))

julia> predictions = predict(T,y)
50304×1 Matrix{SymbolicTransformer.WrappedTransformer.Prediction}:
 Prediction(26.35% " 5", unembed(" 5") ⋅ (T * embed(","))
 Prediction(24.51% " 4", unembed(" 4") ⋅ (T * embed(","))
 Prediction(6.75% " 3", unembed(" 3") ⋅ (T * embed(","))
 Prediction(6.37% " 6", unembed(" 6") ⋅ (T * embed(","))

```

## Current task

trying to have the expression from predict be re-runnable. The expression is now like `:(unembed(" 5") ⋅ (T * embed(","))` but that is not runnable, embed/unembed required T 
as an argument, the ⋅ operation wasn't defined for HGFResidual, and it assumes the transformer is named T. I've added embed/unembed methods that use a global from
the WrappedTransformer module to use the most recently defined PromptedTransformer, but there is still work to be done around values passed as vectors or not.

[![Build Status](https://github.com/prior-technology/SymbolicTransformer/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/prior-technology/SymbolicTransformer/actions/workflows/CI.yml?query=branch%3Amain)
