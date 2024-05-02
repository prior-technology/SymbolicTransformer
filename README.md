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

`WrappedTransformer` represents the results of calculations in types like `Residual`. These include an expression which tracks the origin of the associated result. 

`PromptedTransformer` represents a specific transformer algorithm with prompt text. This acts on a residual vector using the `*` operation to run the internal blocks, returning the residual vector in the last position of the output layer (i.e. excluding input and output embedding layers). 

`predict` is a function which runs the model and calculates logits and probabilities for all tokens, returning each as a Residual which includes an expression which should perform a similar calculation (returning only logits since probabilities depends on all logits for other tokens)

`embed` tokenizes the supplied string and returns a Vector of Residual based on the corresponding entries in the embedding matrix of the transformer. If a transformer is 
not specified the last one defined is used.

`unembed` tokenizes the supplied string and returns a Vector of Residual based on the corresponding entries in the output embedding matrix of the transformer. These are stored as row-vectors in a vector of Residual . 



```julia-repl

julia> T = prompt(model, encoder, "1, 2, 3, 4")
PromptedTransformer(Transformers.HuggingFace.HGFGPTNeoXModel, GPT2TextEncoder, "1, 2, 3, 4")

julia> r = first(embed(T, ","))
Residual(",", embed(","))

julia> y = T * r
Residual("1, 2, 3, 4,", T * embed(","))

julia> predictions = predict(T,y)
50304×1 Matrix{SymbolicTransformer.WrappedTransformer.Prediction}:
 Prediction(26.35% " 5", unembed(" 5") ⋅ (T * embed(","))
 Prediction(24.51% " 4", unembed(" 4") ⋅ (T * embed(","))
 Prediction(6.75% " 3", unembed(" 3") ⋅ (T * embed(","))
 Prediction(6.37% " 6", unembed(" 6") ⋅ (T * embed(","))

```
## Expressions

Many of the types added include an expression which shows how that result was calculated. Expressions like  `(unembed(" 5") ⋅ (T * embed(","))` are runnable but depend on having a PromptedTransformer named T, and the embed/unembed variables refer to this from a global variable which tracks the most recently defined PromptedTransformer.

## Current task

Add an expand function or rule which replaces transformer with the blocks it contains. This is working as far as expanding the full algorithm into a term for each block. The next step is to use an equivalence from Layer Normalise applied to a sum of 
terms to a sum of layer normalised terms. This is returning results which are close, but the difference is greater than expected from rounding errors. 

$$<x, LN (a + b)> = \frac{\sqrt{N}}{\sqrt{|c(a+b)|^2 + N \epsilon} } <x,c(a)> + <x, c(b)> $$

The derivation of this relationship is in reexamine_layer_norm.ipynb, but did not include applying the affine transformation. I think the LN used in Transformer Lens may be different to that from Transformers.jl. I'm currently trying to retrace my steps on this.

This equivalence is also important for the next step, decomposing a Block operation into contributions from Attention and from the Dense layer, from which it will hopefully be possible to see which earlier tokens affected the transformer output.

### Splitting transformer into blocks

The transformer is split into blocks. Currently the block operation includes summing with the residual. This should be changed to keep the sum seperate.

### Splitting predictions

It should be possible to calculate how much each layer and the original embedding contribute to confidence of a specific prediction. Within a block, split between attention and FF layers. And within attention layers which previous positions. 


[![Build Status](https://github.com/prior-technology/SymbolicTransformer/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/prior-technology/SymbolicTransformer/actions/workflows/CI.yml?query=branch%3Amain)
