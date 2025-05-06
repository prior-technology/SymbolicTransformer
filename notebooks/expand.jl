### A Pluto.jl notebook ###
# v0.19.39

using Markdown
using InteractiveUtils

# ╔═╡ a45c14f4-5d78-451b-b63b-eb8db8ce2bdd
using SymbolicTransformer

# ╔═╡ af26023a-af51-4ba2-aff7-f551ecfa6082
using Test

# ╔═╡ fa8d67a6-4007-4f21-b99a-3ba79645e379
using Transformers.HuggingFace

# ╔═╡ bfca0533-009c-47fb-8c80-95d1322b41bb
using Transformers.TextEncoders

# ╔═╡ 05ca19cd-2ee1-4dfc-8d20-c13d4382c709
using SymbolicTransformer.WrappedTransformer

# ╔═╡ 8b9df7d5-3b26-4924-85e0-87f92c2bbded
using TextEncodeBase

# ╔═╡ f1469f34-66bc-4f0e-8f51-7e21801e4faa
using LinearAlgebra

# ╔═╡ 94ae5378-0326-4cbf-a493-c640da95f50a
const encoder = hgf"EleutherAI/pythia-14m:tokenizer"

# ╔═╡ 3670657d-111e-4815-bd63-8afc73c29225
const model = hgf"EleutherAI/pythia-14m:forcausallm"

# ╔═╡ Cell order:
# ╠═a45c14f4-5d78-451b-b63b-eb8db8ce2bdd
# ╠═af26023a-af51-4ba2-aff7-f551ecfa6082
# ╠═fa8d67a6-4007-4f21-b99a-3ba79645e379
# ╠═bfca0533-009c-47fb-8c80-95d1322b41bb
# ╠═05ca19cd-2ee1-4dfc-8d20-c13d4382c709
# ╠═8b9df7d5-3b26-4924-85e0-87f92c2bbded
# ╠═f1469f34-66bc-4f0e-8f51-7e21801e4faa
# ╠═94ae5378-0326-4cbf-a493-c640da95f50a
# ╠═3670657d-111e-4815-bd63-8afc73c29225
