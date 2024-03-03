
using Transformers.HuggingFace
using SymbolicTransformer
encoder, model = hgf"EleutherAI/pythia-70m-deduped"

@testset "prompt" begin
    p = WrappedTransformer.prompt(model, encoder, "Hello, world!")
    @test p.prompt == "Hello, world!"
end
