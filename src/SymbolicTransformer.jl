module SymbolicTransformer

using Symbolics

include("SymbolCreator.jl")
include("LayerNormalization.jl")

#abstract type Residual <:AbstractVector  end
# "returns an expression which represents the operation of a transformer on a residual vector.
# The first expansion is adding the output from each layer to the input residual, and applying layer normalization."
# function expand(T::Transformer)
#     @variables layer_output[1:T.n_layers] :: Array{}
# end
struct Token
    id
    text
    position
    vector
end

struct Transformer
    blocks
end

struct TransformerBlock
    attention
    feedforward
    norm1
    norm2
end

#pythia-70 sizes
d_model=512
n_heads=8
d_head=d_model/n_heads

struct Residual
    vector
end

"Return a symbolic expression that shows that T acts on x to produce a new residual y"
function transform_expression(T::Transformer, x::Residual)
    y = T(x)
end
    
function expand

end