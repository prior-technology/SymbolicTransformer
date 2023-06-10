using SymbolicTransformer
using LinearAlgebra

include("../data/probe_token.jl")

include("../data/pre_norm.jl")

bias = 0.8328

final_residual = LN(pre_norm)

id = ones(512)
beta = zeros(512)

f = layer_norm(id, beta, 1e-5)

final_residual = f(pre_norm)
logit = (probe_token ⋅ final_residual) + bias

print(logit)