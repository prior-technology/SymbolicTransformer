# Goal

This folder will support an environment with Symbolic Transformer in Julia and Transformer Lens in Python. Symbolic Transformer will have an interface to support concrete weights and residual vectors from external transformer implementations. This folder will use Transformer Lens to provide these vectors.

## Setup

This is the start of an attempt to include [Transformer Lens](https://neelnanda-io.github.io/TransformerLens/) in this project, to 
be used to populate weights and validate implementation.

Transformer-lens installs more easily once pytorch is preinstalled

Attempted to set up an env using miniconda, tried to update CondaPkg.toml to match. Running notebook resulted in torch packages getting removed from existing environment.

Setup python environment just using conda for torch and pip to install transformer-lens and juliacall. Still not finding the right way to pass ActivationCache type from 
Transformer Lens to Julia - I'll just have to keep copy-pasting vectors for now.
