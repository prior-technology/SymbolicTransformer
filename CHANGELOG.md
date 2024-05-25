# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0-DEV] - 2014-05-31

This version is implemented primarily by wrapping the Transformers.j
### Added

  - Applying a PromptedTransformer to a Residual results in a collection of Predictions representing the LM's prediction of the likelihood of each possible subsequent token
  - Running expand on a Prediction breaks the prediction based on the contribution of each layer to the prediction
