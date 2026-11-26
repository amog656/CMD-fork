# Notes
Here are the splits that we used for training/evaluating aSAMt models on mdCATH data. See the aSAM publication for details on how the splits were created. The names of the system follow this pattern:

`${CATH_DOMAIN_ID}.${TEMPERATURE}`

where `$TEMPERATURE` is the temperature of the MD simulation for that system.

## Autoencoder training
* First stage
  * training: `train.max_len_320.txt`
    * note: this split contains only training systems containing a protein domain with length <= 320. See `train.txt` file for a list of all training systems.
  * validation: `val.txt`
  * test: `test.txt`
## Diffusion model training
* First stage:
  * training: `train.max_len_320.fix.txt`
    * note: this list is derived from `train.max_len_320.txt`, but here we excluded all systems for 4j2nC00, 1b5fC00, 3hshE00, 1ow4A00, 3h33A00, 3vhoA00 and 5e99H01, because of errors when encoding their trajectories with our encoder network.
  * validation: `val.txt`
  * test: `test.txt`