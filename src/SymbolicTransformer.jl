module SymbolicTransformer

using Symbolics

include("SymbolCreator.jl")

export LN

"Expectation or Mean of a vector"
μ(v) = sum(v)/size(v,1)

"Center vector to have mean 0"
center(x) = x .- μ(x)

"This implementation of  Layer Normalization is based on LayerNormPre in 
Transformer Lens. Gives same result for specific example checked."
#original 
#x = x - x.mean(axis=-1, keepdim=True)
#scale = x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
#return x/scale
function LN(v, ϵ = 1e-5)
    top = (v .- μ(v))
    scale = (μ((v .- μ(v)).^2) + ϵ).^(-0.5)    
    top .* scale
end

#abstract type Residual <:AbstractVector  end
# "returns an expression which represents the operation of a transformer on a residual vector.
# The first expansion is adding the output from each layer to the input residual, and applying layer normalization."
# function expand(T::Transformer)
#     @variables layer_output[1:T.n_layers] :: Array{}
# end


end