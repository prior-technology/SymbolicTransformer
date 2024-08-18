module SymbolicTransformer

include("LayerNormalization.jl")

abstract type Prediction end

"""
    Returned from prompt, holds Transformer T, embedding weights W_E, and unembedding weights W_U
"""
struct PromptedTransformer
    T    
    W_E 
    W_U
    encode
    tokenize
    embed
end


function expand(expr::Expr)
    return expr
end


    
end