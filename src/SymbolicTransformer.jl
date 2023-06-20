module SymbolicTransformer

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
struct Attention_Params
    W_Qs :: Array{Matrix}
    W_Ks :: Array{Matrix}
    W_Vs :: Array{Matrix}
    W_Os :: Array{Matrix}
    b_Q :: Vector
    b_K :: Vector
    b_V :: Vector

end

function Base.show(io::IO, t::Token)
    print(io, "$(t.text)_{$(position)}")
end


function forward(transformer::TransformerBlock, residuals)
    #assume that residuals are already embedded and positioned

    for block in transformer.blocks
        residuals = residuals + forward(block, residuals)
    end
end

function batch_layer_normalize(residuals)
    for residual in residuals
        residual = LN(residual)
    end
end
function forward(transformerBlock::TransformerBlock , residuals)
    block_input = batch_layer_normalize(residuals)
   
    attention_out = attention(transformerBlock.attention, block_input)
    mlp_in = batch_layer_normalize(attention_out)
    mlp_out = mlp(transformerBlock.feedforward, mlp_in)
    return mlp_out

end

function attention(attention, attention_in)
    for h = 1:n_heads
        attention_out = attention_out + attention_head(attention, attention_in, h)
    end
    return attention_out
end

function attention_head(attention::Attention_Params, residual, h)
    W_Q = attention.W_Qs[h]
    q = ( W_Q * residual ) + attention.W_Q_bias
    W_K = attention.W_Ks[h]
    k = ( W_K * residual ) + attention.W_K_bias
    W_V = attention.W_Vs[h]
    v = ( W_V * residual ) + attention.W_V_bias

    q = apply_rotary(q)
    k = apply_rotary(k)
    #TODO: continue- calculate attention scores
    attention_matrix = attention_scores(q, k)
    softmax
    O = attention.W_O * attention_in
    return O
end

function attention_scores(h, q, k)
    attention_scores = zeros(seq_len, seq_len)
    for pos = 1:seq_len
        for pos2 = 1:pos
            q = q[pos, h]
            k = k[pos2, h]
            attention_scores[pos, pos2] = q * k
        end
    end
    return attention_scores ./ sqrt(d_head) 
end
end