module SymbolicTransformer

include("LayerNormalization.jl")
include("VectorTransformer.jl")


struct Operation 
    forward :: Function
    expression :: Expr
    label :: AbstractString
end
struct Residual
    vector :: AbstractVector
    expression :: Expr
    label :: AbstractString
end

    
end