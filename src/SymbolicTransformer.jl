module SymbolicTransformer

export LN

μ(v) = sum(v)/size(v,1)

function LN(v, ϵ = 1e-5)
    top = (v .- μ(v))
    scale = (μ(v.^2) + ϵ).^(-0.5)
    top .* scale
end

end
