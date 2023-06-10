module SymbolicTransformer

export LN, layer_norm

"Expectation or Mean of a vector"
μ(v) = sum(v)/size(v,1)

"Center vector to have mean 0"
center(x) = x .- μ(x)

"2-norm or length of a vector"
norm(x) = sqrt(sum(x.^2))

"Returns a function which performs an Affine transformation"
affine(γ, β) = x -> γ .* x .+ β

"Returns a function which performs a non-linear normalization transformation"
u(ϵ = 1e-5) = 
    function u_ϵ(x) 
        n = size(x,1)
        x ./ (( 1 / n ).* norm(x) + ϵ)
    end

"This implementation of Layer Normalization is based on the description in https://www.alignmentforum.org/posts/jfG6vdJZCwTQmG7kb/re-examining-layernorm"
function layer_norm(γ, β, ϵ=1e-5)
    affine(γ, β) ∘ u(ϵ) ∘ center
end

"This implementation of  Layer Normalization is based on LayerNormPre in Transformer Lens"
function LN(v, ϵ = 1e-5)
    top = (v .- μ(v))
    scale = (μ(v.^2) + ϵ).^(-0.5)
    top .* scale
end

end
