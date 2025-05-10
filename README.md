# SymbolicTransformer

This project works through some ideas around language model interpretability through Julia.

# Motivation

Language models consist of billions of numbers which are combined in a complicated pattern with other blocks of numbers which represent meaningful text to generate more numbers representing more text. The goal is to name symbols and operations representing different stages of this calculation so they can be reasoned about further

## Short Term Goal

To generate values representing inputs and outputs to a transformer language model which can be combined to perform the model's processing, and allow terms to be expanded to show intermediate steps.

Prompting a model should return a type containing:
 - `context`\
   This encapsulates a specific language model and tokenizer, and any other model specific data required.   
 - `setup`\
   This expression defines symbols used in the expression depending on prompt text and any subsequent manipulation. It assumes context is available through a symbol ctx
 - `transform`\
   This should be a simple expression showing the operation of the transformer on a vector in the embedding space.
 - `interpretation`\
   This expression defines the output of the model (e.g. prediction of next token) using symbols from the previous stages.


```julia-repl

julia> using Transformers.HuggingFace

julia> using SymbolicTransformer

julia> context = init_context(hgf"EleutherAI/pythia-70m-deduped")
WrappedTransformer

julia> T = prompt(context, "The capital of Ireland is")
Transformation(
       :(T = ctx.transformer.embed("the capital of Ireland is")),
       :(y = T * ctx.embed("is")),
       :empty       
)

julia> generate(T)

Transformation(
       :(T = ctx.transformer.embed("the capital of Ireland is")),
       :(y = T * E[is]),
       quote
              logits = ctx.embed_out(y)
              next_token_logits = logits[:, -1, :]
              next_token_id = torch.argmax(next_token_logits, dim=-1)  
              next_token = ctx.tokenizer.decode(next_token_id)
              return next_token
       end
)

julia> expand(context, :(T * E[is]))
:(L4 * (L3 * (L2 * (L1 * E[is]))))

```

## Medium Term Goal

To use attribution, gradients and estimation to identify and extract features relevant to a particular calculation through the model, and to neglect terms
which have lower relevance to a particular calculation.

```julia

julia> L1 * r
Key1 + V2 + r 

```

# Plans and Progress

This refactor branch is intended to enable more readable expressions with clearer separation between the trained model, 
the context, and between operations and vectors.

## Problem

The representation of how the different blocks of a transformer contribute to a particular prediction looks like:

```Prediction(0.01% l=0.54 unembed(" 5") ⋅ (expand(T, T * embed(",")))[3])```

This is difficult to parse but still loses too much relevant information required for the next stage, where we seperate contributions 
from attention and MLP layers of that block.

Instead of trying to encapsulate everything in a prediction type can we split some of this out using different operations.

In words, the summary above says: "In a specific context when a prompted transformer acts on the embedding vector for token ",", and considering the
predicted likelihood that the next token is " 5", the output of the 3rd block contributes 0.54 to resulting logit, which
represents a contribution of 0.01% to the likelihood."

The significant challenge in readability is the (expand(T, T * embed(",")))[3] expression, which represents the output of the third block when PromptedTransformer T acts on Residual embed(","). Alternative ways to write this could be

 - block(3, :(T * embed(",")))
 - expand(T, T * embed(","))[3]
 - transformation = :(T * embed(","))
   transformation.block[3]

Can we center the Residual vectors and operations rather than the prediction?

Let T be a bare trained transformer model (i.e. without embedding/unembedding layers)
E be the embedding layer
U the unembedding layer
A Residual is a vector in the residual space of a transformer
A Prompted Transformer is combination of Transformer, Text and corresponding Residual vectors.
A Transformation is the operation of a Prompted Transformer on a sequence of Residuals

Prediction is the probability next token will be y given output residual x. This depends on all unembedding vectors (so softmax can be calculated)

Expand(expression) tries to replace the top layer of an expression with its components how each block contributes to x. It depends on PT and input residual
Prompted Transformer T("1,2,3,4") acting on Residual embed(",") is a process/operation/transformation which could be
referred to and analysed.

So `T(1,2,3,4) ∘ embed(",")`would return the output residual in the last position.

Transformation is represented as an expression `transformation = :T(1,2,3,4) ∘ embed(",")`

block(3) references the 3rd block, block(1) and block(last) are also possible

internal_vector = output(transformation,block(3))

## High Level Concepts

Residual : a vector in the residual space of a particular Transformer
Map : maps one residual to another, not necessarily linear or invertible
Unembed : maps a residual to a logit
Prediction : given model with tokenizer, 
Transformer maps a sequence of residuals to another

PromptedTransformer maps one residual to another.

Reference - identifies a block, position within a block,head, etc.

## Previous Description

I'm aiming to see the flow through using Transformers.jl with Pythia/GPTNeo-X models. Later it should be possible to abstract out the logic which doesn't directly depend on a specific implementation. Earlier work started to rewrite the algorithm from scratch, and earlier again focussed on abstract operations without specific implementations.

`WrappedTransformer` represents the results of calculations in types like `Residual`. These include an expression which tracks the origin of the associated result. 

`PromptedTransformer` represents a specific transformer algorithm with prompt text. This acts on a residual vector using the `*` operation to run the internal blocks, returning the residual vector in the last position of the output layer (i.e. excluding input and output embedding layers). 

`predict` is a function which runs the model and calculates logits and probabilities for all tokens, returning each as a Residual which includes an expression which should perform a similar calculation (returning only logits since probabilities depends on all logits for other tokens)

`embed` tokenizes the supplied string and returns a Vector of Residual based on the corresponding entries in the embedding matrix of the transformer. If a transformer is 
not specified the last one defined is used.

`unembed` tokenizes the supplied string and returns a Vector of Residual based on the corresponding entries in the output embedding matrix of the transformer. These are stored as row-vectors in a vector of Residual . 



```julia-repl

julia> using Transformers.HuggingFace
       using SymbolicTransformer;
       using SymbolicTransformer.WrappedTransformer;
       const encoder = hgf"EleutherAI/pythia-14m:tokenizer"
       const model = hgf"EleutherAI/pythia-14m:forcausallm"

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
The expand command seperates contributions to the logit from each of the 7 transformer block and from the embedding residual.

```julia-repl
julia> expand(T, predictions[1], r)
8-element Vector{SymbolicTransformer.WrappedTransformer.PredictionTerm}:
 Prediction(-0.06% l=-3.79 unembed(" 5") ⋅ center(embed(",")))
 Prediction(0.06% l=3.31 unembed(" 5") ⋅ (expand(T, T * embed(",")))[2])
 Prediction(0.01% l=0.54 unembed(" 5") ⋅ (expand(T, T * embed(",")))[3])
 Prediction(0.31% l=18.36 unembed(" 5") ⋅ (expand(T, T * embed(",")))[4])
 Prediction(-0.02% l=-1.05 unembed(" 5") ⋅ (expand(T, T * embed(",")))[5])
 Prediction(0.57% l=34.38 unembed(" 5") ⋅ (expand(T, T * embed(",")))[6])
 Prediction(14.21% l=852.58 unembed(" 5") ⋅ (expand(T, T * embed(",")))[7])
 Prediction(11.55% l=693.33 unembed(" 5") ⋅ T.ln.β)

```

## Expressions

Many of the types added include an expression which shows how that result was calculated. Expressions like  `(unembed(" 5") ⋅ (T * embed(","))` are runnable but depend on having a PromptedTransformer named T, and the embed/unembed functions refer to this from a global variable which tracks the most recently defined PromptedTransformer.

[![Build Status](https://github.com/prior-technology/SymbolicTransformer/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/prior-technology/SymbolicTransformer/actions/workflows/CI.yml?query=branch%3Amain)
