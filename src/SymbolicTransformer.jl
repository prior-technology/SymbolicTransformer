module SymbolicTransformer

include("LayerNormalization.jl")
include("VectorTransformer.jl")
include("WrappedTransformer.jl")

abstract type Operation end
#     expression :: Expr
#     label :: AbstractString
# end

abstract type Residual end
#     vector :: AbstractVector
#     expression :: Expr
#     label :: AbstractString
# end

    
end