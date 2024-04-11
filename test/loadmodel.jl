module TestData
using Transformers
using Transformers.HuggingFace
using Transformers.TextEncoders

const encoder = hgf"EleutherAI/pythia-14m:tokenizer"
const model = hgf"EleutherAI/pythia-14m:forcausallm"

function get_encoder()
    return encoder
end
function get_model()
    return model
end

function get_both()
    return model, encoder
end

end
