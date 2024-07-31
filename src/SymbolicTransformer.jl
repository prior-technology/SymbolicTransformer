module SymbolicTransformer

include("LayerNormalization.jl")

abstract type Operation end
#     expression :: Expr
#     label :: AbstractString
# end

abstract type Residual end
#     vector :: AbstractVector
#     expression :: Expr
#     label :: AbstractString
# end
abstract type Prediction end


function expand(expr::Expr)
    return expr
end


    
end