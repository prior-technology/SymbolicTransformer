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

# ╔═╡ 1b5fae88-5d8d-4ed0-a76d-164de1e37300
md"Define a PromptedTranformer T. This wraps the Transformers.jl HuggingFace model along with a string which is prepended to vectors the model acts on."

# ╔═╡ d8d9fe9e-e699-495c-a2a8-7ecf55dd58d4
T = prompt(model, encoder, "1, 2, 3, 4")

# ╔═╡ a90048ef-aa3a-478e-a2cb-da3fc1f8c1f0
input = embed(T, ",")

# ╔═╡ b674f0b8-4ae5-41a7-a005-7b9d605b48f4
md"input is an array of HGFResidual, which wraps the residual vector from embedding each token. The * operator applies the T operator to the input by runining the combined input through the model. "

# ╔═╡ 117b8732-4951-44a6-a641-8760da771997
y = T * input

# ╔═╡ 60d5ebfb-2355-4689-83dc-a9de6318e485
The result y is another HGF resdidual representing the result of the language model before the unembedding layer.

# ╔═╡ 7f622c65-2c04-4911-ad70-a0a728357e14
predictions = predict(T,first(y))

# ╔═╡ b3850e57-aed1-4d32-86f6-8907d573959a
md"The predict function calculates logits and the scale factors to normalise logits using softmax so they can be interpreted as percentages. This is the core function of a language model - predicting tokens based on previous context. The training objective for the core model is based on the loss from comparing the actual tokens to predicted ones."

# ╔═╡ 86840363-12e3-45e2-aae5-970f208ff592
@show predictions[1]

# ╔═╡ db8a014f-9b9a-49ed-8186-ebfb6aaff384
md"The Prediction struct includes token_id, logits, probability of next token, and an indicative expression showing how the logit is calculated. The next two cells show the calculation repeated by following the expression resulting in an equivalent logit value `(1581.6716f0)`"

# ╔═╡ e71356a7-c0f7-4b7e-bc7e-6a5852d01d2a
y2 = first(unembed(" 5") ⋅ (T * embed(",")))

# ╔═╡ f0ab8d2b-73d6-4b06-9fbf-692c0258ffaa
y2.vector

# ╔═╡ 726c5255-be7e-42d3-b81c-f9847cb251be
md"What I want here is to replace T in the expression above with entries representing each block of the transformer. Based on analysis elsewhere this should be possible using the identity ``<x, LN (a + b)> =  \frac{\sqrt{N}}{\sqrt{|c(a+b)|^2 + N \epsilon} } (<x,c(a)> + <x, c(b)>) ``

For now, extract_blocks is implemented to split T into a Layer Normalization Transformers.jl layer, and an array of blocks, the results of which are summed before LN is applied"

# ╔═╡ 0e669263-23d4-4ec9-a621-bcc561146e99
(ln, prompt_blocks) = extract_blocks(T)

# ╔═╡ 32939b08-668b-4082-8158-9187095b3d3d
block_deltas=[]

# ╔═╡ 838c32a9-2996-4b99-9aa7-b6a2f949333e


# ╔═╡ 24fb2b30-9d71-41ad-8e5a-e28be34f469f
(input_residual, total_residual) = let
	input_residual = total_residual = embed(",")
	for block in prompt_blocks
		this_block_residual = block * total_residual 
		push!(block_deltas, this_block_residual)
		total_residual  = total_residual + this_block_residual
	end
	(input_residual, total_residual)
end

# ╔═╡ dea9553d-dae0-43eb-be98-2c354d3c13e0
md"""
 `block_deltas` now contains a vector of vectors of HgfResidual and `total_residual` has the final residual before layer normalization.

Calculating the factor used:

``\frac{\sqrt{N}}{\sqrt{|c(a+b)|^2 + N \epsilon} }``
"""


# ╔═╡ 699bf570-d3b7-45c6-8ac3-eb85db71710d
#Expectation or Mean of a vector
μ(v) = sum(v)/size(v,1)

# ╔═╡ 1f521ea1-27ad-4ea5-b3bb-cf3f97515feb
#Center vector to have mean 0
center(x) = x .- μ(x)

# ╔═╡ 2727f869-0053-40bb-a9bc-72490158907c
N = length(input_residual[1].vector)

# ╔═╡ f3f63eb9-1ee9-4280-941d-e24ced7125ea
c_ab = (center(first(total_residual).vector))

# ╔═╡ a12be89b-372f-4c17-bb26-e3304fcb7bfe
c_ab2 = LinearAlgebra.norm(c_ab, 1)

# ╔═╡ 0df930fb-3e03-4e14-9c1f-fd0295acd9bf
Nϵ = N * ln.ϵ

# ╔═╡ 46529760-c667-476d-9ade-0b94cbb164de
factor = (sqrt(N)/(c_ab2 + Nϵ))

# ╔═╡ a052a798-9473-416e-92cb-ee169ae78a30
md"""So again using 

``<x, LN (a + b)> =  \frac{\sqrt{N}}{\sqrt{|c(a+b)|^2 + N \epsilon} } (<x,c(a)> + <x, c(b)>) ``

we have predictions[1] =``<\textrm{unembed(" 5")} , (T * \textrm{embed(",")}) >``, so 

``<\textrm{unembed(" 5")} , (T * \textrm{embed(",")}) >
= <\textrm{unembed(" 5")} , LN (\sum{\textrm{block_delta}} )> ``
``= \frac{\sqrt{N}}{\sqrt{|c(total residual)|^2 + N \epsilon} } (<x,\textrm{c(block1_delta)}> + <x, \textrm{c(block2_delta)}> ...) ``

"""

# ╔═╡ d82f0ae6-b449-4b9c-966d-b0e01bb2122d
x = predictions[1]

# ╔═╡ 27e5da4c-5c7f-4d3a-82c0-e4ccfa64647d
contributions = []

# ╔═╡ 96cffeea-4e1f-4e19-835b-00cb96fa34b2
#contribution from embedding layer
push!(contributions, factor * (unembed(" 5") ⋅ center(first(embed(",")).vector)))

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
# ╟─1b5fae88-5d8d-4ed0-a76d-164de1e37300
# ╠═d8d9fe9e-e699-495c-a2a8-7ecf55dd58d4
# ╠═a90048ef-aa3a-478e-a2cb-da3fc1f8c1f0
# ╠═b674f0b8-4ae5-41a7-a005-7b9d605b48f4
# ╠═117b8732-4951-44a6-a641-8760da771997
# ╠═60d5ebfb-2355-4689-83dc-a9de6318e485
# ╠═7f622c65-2c04-4911-ad70-a0a728357e14
# ╟─b3850e57-aed1-4d32-86f6-8907d573959a
# ╠═86840363-12e3-45e2-aae5-970f208ff592
# ╟─db8a014f-9b9a-49ed-8186-ebfb6aaff384
# ╟─e71356a7-c0f7-4b7e-bc7e-6a5852d01d2a
# ╠═f0ab8d2b-73d6-4b06-9fbf-692c0258ffaa
# ╠═726c5255-be7e-42d3-b81c-f9847cb251be
# ╠═0e669263-23d4-4ec9-a621-bcc561146e99
# ╠═32939b08-668b-4082-8158-9187095b3d3d
# ╠═838c32a9-2996-4b99-9aa7-b6a2f949333e
# ╠═24fb2b30-9d71-41ad-8e5a-e28be34f469f
# ╠═dea9553d-dae0-43eb-be98-2c354d3c13e0
# ╠═699bf570-d3b7-45c6-8ac3-eb85db71710d
# ╠═1f521ea1-27ad-4ea5-b3bb-cf3f97515feb
# ╠═2727f869-0053-40bb-a9bc-72490158907c
# ╠═f3f63eb9-1ee9-4280-941d-e24ced7125ea
# ╠═a12be89b-372f-4c17-bb26-e3304fcb7bfe
# ╠═0df930fb-3e03-4e14-9c1f-fd0295acd9bf
# ╠═46529760-c667-476d-9ade-0b94cbb164de
# ╠═a052a798-9473-416e-92cb-ee169ae78a30
# ╠═d82f0ae6-b449-4b9c-966d-b0e01bb2122d
# ╠═27e5da4c-5c7f-4d3a-82c0-e4ccfa64647d
# ╠═96cffeea-4e1f-4e19-835b-00cb96fa34b2
