# Notes
Here are the splits that we used for training/evaluating aSAMc models on ATLAS data. We used the same splits of [AlphaFlow](https://github.com/bjing2016/alphaflow).
## Autoencoder training
* First stage
  * training: `training.max_len_320.txt`
  * validation: `val.max_len_500.txt`
* Second stage:
  * training: `training.max_len_500.txt`
  * validation: `val.max_len_500.txt`
  * test: `test.txt`
## Diffusion model training
* First stage:
  * training: `training.max_len_500.txt`
  * validation: `val.max_len_500.txt`
  * test: `test.txt`