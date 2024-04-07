### A Pluto.jl notebook ###
# v0.19.39

using Markdown
using InteractiveUtils

# ╔═╡ 41be49ce-5b8b-4ad7-8979-2ab371daeb24
using Pkg

# ╔═╡ b79c93ae-f3f7-43a4-b65e-d7a2d60bbada
Pkg.activate(joinpath(@__DIR__, "..") )

# ╔═╡ fa8d67a6-4007-4f21-b99a-3ba79645e379
# ╠═╡ show_logs = false
using Transformers.HuggingFace

# ╔═╡ bfca0533-009c-47fb-8c80-95d1322b41bb
using Transformers.TextEncoders

# ╔═╡ 05ca19cd-2ee1-4dfc-8d20-c13d4382c709
# ╠═╡ show_logs = false
using SymbolicTransformer.WrappedTransformer

# ╔═╡ 8b9df7d5-3b26-4924-85e0-87f92c2bbded
using TextEncodeBase

# ╔═╡ f1469f34-66bc-4f0e-8f51-7e21801e4faa
using LinearAlgebra

# ╔═╡ 50474bad-5fb4-49fd-8e30-05c40fceb0af
using SymbolicTransformer

# ╔═╡ 1dc647c4-7e21-421f-9a24-85b18f74ce7e
md"Using julia environment from this repo because the SymbolicTransformer package is as yet unpublished"

# ╔═╡ 94ae5378-0326-4cbf-a493-c640da95f50a
# ╠═╡ show_logs = false
const encoder = hgf"EleutherAI/pythia-14m:tokenizer"

# ╔═╡ 3670657d-111e-4815-bd63-8afc73c29225
const model = hgf"EleutherAI/pythia-14m:forcausallm"

# ╔═╡ d8d9fe9e-e699-495c-a2a8-7ecf55dd58d4
T = prompt(model, encoder, "1, 2, 3, 4")

# ╔═╡ a90048ef-aa3a-478e-a2cb-da3fc1f8c1f0
input = embed(T, ",")

# ╔═╡ 117b8732-4951-44a6-a641-8760da771997
y = T * input

# ╔═╡ 7f622c65-2c04-4911-ad70-a0a728357e14
predictions = predict(T,first(y))

# ╔═╡ a4d5a6e2-7e95-4e1c-8a4e-7d8a8e5437e3
(predictions[1].logit, predictions[1].normalization_constant)

# ╔═╡ e71356a7-c0f7-4b7e-bc7e-6a5852d01d2a
y2 = unembed(" 5") ⋅ (T * embed(","))

# ╔═╡ f0ab8d2b-73d6-4b06-9fbf-692c0258ffaa
first(y2).vector

# ╔═╡ 726c5255-be7e-42d3-b81c-f9847cb251be
md"What I want here is to replace T in the expression above with entries representing each block of the transformer. Based on analysis elsewhere this should be possible using the identity \$\$<x, LN (a + b)> = \sqrt{N} \frac{<x,c(a)> + <x, c(b)>}{\sqrt{|c(a+b)|^2 + N \epsilon} } \$\$"

# ╔═╡ 0e669263-23d4-4ec9-a621-bcc561146e99
residuals = let
	input = (; token=T.tokens)
    v = T.embed_layer(input)
    v.hidden_state
end

# ╔═╡ 24fb2b30-9d71-41ad-8e5a-e28be34f469f
block1 = WrappedTransformer.PromptedTransformerBlock(T.model.decoder.layers[1][1], residuals, :(T.Block[1]))

# ╔═╡ 1015b385-63de-445d-ad27-2625ce94f884
block1_input = hcat([r.vector for r in embed(",")]...)

# ╔═╡ fa97f02c-3a6e-498d-a0be-bc636808126f
hidden_state = hcat(block1.prompt_residuals, block1_input)

# ╔═╡ 95485a9e-4c63-408f-b257-d354e249c94f
block2_input = block1.block((; hidden_state=hidden_state))

# ╔═╡ Cell order:
# ╟─1dc647c4-7e21-421f-9a24-85b18f74ce7e
# ╠═41be49ce-5b8b-4ad7-8979-2ab371daeb24
# ╠═b79c93ae-f3f7-43a4-b65e-d7a2d60bbada
# ╠═fa8d67a6-4007-4f21-b99a-3ba79645e379
# ╠═bfca0533-009c-47fb-8c80-95d1322b41bb
# ╠═05ca19cd-2ee1-4dfc-8d20-c13d4382c709
# ╠═8b9df7d5-3b26-4924-85e0-87f92c2bbded
# ╠═f1469f34-66bc-4f0e-8f51-7e21801e4faa
# ╠═50474bad-5fb4-49fd-8e30-05c40fceb0af
# ╠═94ae5378-0326-4cbf-a493-c640da95f50a
# ╠═3670657d-111e-4815-bd63-8afc73c29225
# ╠═d8d9fe9e-e699-495c-a2a8-7ecf55dd58d4
# ╠═a90048ef-aa3a-478e-a2cb-da3fc1f8c1f0
# ╠═117b8732-4951-44a6-a641-8760da771997
# ╠═7f622c65-2c04-4911-ad70-a0a728357e14
# ╠═a4d5a6e2-7e95-4e1c-8a4e-7d8a8e5437e3
# ╠═e71356a7-c0f7-4b7e-bc7e-6a5852d01d2a
# ╠═f0ab8d2b-73d6-4b06-9fbf-692c0258ffaa
# ╟─726c5255-be7e-42d3-b81c-f9847cb251be
# ╠═0e669263-23d4-4ec9-a621-bcc561146e99
# ╠═24fb2b30-9d71-41ad-8e5a-e28be34f469f
# ╠═1015b385-63de-445d-ad27-2625ce94f884
# ╠═fa97f02c-3a6e-498d-a0be-bc636808126f
# ╠═95485a9e-4c63-408f-b257-d354e249c94f
