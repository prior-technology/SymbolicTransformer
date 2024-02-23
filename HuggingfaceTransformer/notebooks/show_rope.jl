### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 2e74c427-4751-4122-9725-6d3220243795
using Pkg

# ╔═╡ f51a822e-cdbd-11ee-3124-6b772d61a07f
Pkg.develop(path="/home/stevop/repos/SymbolicTransformer/HuggingfaceTransformer")

# ╔═╡ 62e27717-71fc-4723-8331-2503b788cad6
using HuggingfaceTransformer 

# ╔═╡ d7a9abc6-0066-4ee3-b77b-47a89b96c910
(model,tokenizer) = load_gptneox("EleutherAI/pythia-70m-deduped","step3000")

# ╔═╡ d2040c7d-ffe5-4dae-b478-8889816677cb
model.gpt_neox.layers[0].attention.rotary_emb

# ╔═╡ bd1f326d-186e-4c4e-8c7b-3ff2af6d930c
inputs = tokenizer("which which is which", return_tensors="pt")


# ╔═╡ 6cf22d30-4193-46c4-a37a-64fee9790bd4
tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

# ╔═╡ 58e260b1-c88c-424c-9215-743a86bc6af6
tokens = model.generate(input_ids = inputs.input_ids,
		attention_mask=inputs.attention_mask)

# ╔═╡ e1e98aef-e325-4aca-b686-f64fe4a7024f
tokenizer.decode(tokens[0])

# ╔═╡ 2f1d66d1-7330-4c0d-babb-b83b92df43f8
md"### Calculate an embedding vecor for token 534 in 1st and 3rd position"

# ╔═╡ f887266a-a930-4a0d-ba6b-997cdfa2762d
t = model.gpt_neox.embed_in.forward(inputs.input_ids)

# ╔═╡ 49edb6c7-2773-4939-9b80-bb82d97a3278
t.shape

# ╔═╡ 88b84cd1-53f9-46c4-b8bd-1b8510ec296d
t[0,1].equal(t[0,3])

# ╔═╡ 5bf35d3f-71e9-442f-89ef-54a97d9421f1
rope = model.gpt_neox.layers[0].attention.rotary_emb

# ╔═╡ 3c8e9c76-ceb4-4a6d-aad4-b05cac9d7a42
md"""
from GPTNeoXAttention
```python
cos, sin = self.rotary_emb(value, seq_len=seq_len)
```
value is one of query,key,value from hidden_states, and is just used to check device and data type in use. It generates a cache of cosine and sine values based on sequence length

```python
query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
```
uses the cached values above to apply the rotary pattern to q and k attention vectors
"""


# ╔═╡ Cell order:
# ╠═2e74c427-4751-4122-9725-6d3220243795
# ╠═f51a822e-cdbd-11ee-3124-6b772d61a07f
# ╠═62e27717-71fc-4723-8331-2503b788cad6
# ╠═d7a9abc6-0066-4ee3-b77b-47a89b96c910
# ╠═d2040c7d-ffe5-4dae-b478-8889816677cb
# ╠═bd1f326d-186e-4c4e-8c7b-3ff2af6d930c
# ╠═6cf22d30-4193-46c4-a37a-64fee9790bd4
# ╠═58e260b1-c88c-424c-9215-743a86bc6af6
# ╠═e1e98aef-e325-4aca-b686-f64fe4a7024f
# ╠═2f1d66d1-7330-4c0d-babb-b83b92df43f8
# ╠═f887266a-a930-4a0d-ba6b-997cdfa2762d
# ╠═49edb6c7-2773-4939-9b80-bb82d97a3278
# ╠═88b84cd1-53f9-46c4-b8bd-1b8510ec296d
# ╠═5bf35d3f-71e9-442f-89ef-54a97d9421f1
# ╟─3c8e9c76-ceb4-4a6d-aad4-b05cac9d7a42
