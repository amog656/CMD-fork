---
# Doc / guide: https://huggingface.co/docs/hub/model-cards
license: Apache 2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
doi: https://doi.org/10.1101/2025.03.09.642148
language:
- en
tags:
- proteins
- structural ensembles
- molecular dynamics
- thermostability
- generative modeling
- autoencoder
- diffusion model
- transformers
---

# Model Card for aSAM

<!-- Provide a quick summary of what the model is/does. -->

Deep generative model trained on molecular dynamics (MD) data for generating atomistic protein structural ensembles.

## Model Details

aSAM (atomistic Structural Autoencoder Model) is a latent diffusion-based generative model to efficiently sample protein structural ensembles at atomistic resolution. aSAM encodes heavy atoms in a latent space, enabling accurate sampling of both backbone and side-chain torsion angles. By training on the [mdCATH](https://github.com/compsciencelab/mdCATH) dataset, we developed aSAMt, the first known transferable protein ensemble generator conditioned explicitly on an environmental factor, in this case, temperature. This allows the model to capture temperature-dependent ensemble properties as observable in MD simulations.

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** Giacomo Janson, Alexander Jussupow, Michael Feig
- **Funded by:** National Institutes of Health, grant R35 GM126948 (to MF)
- **Shared by:** Feig Lab ([Department of Biochemistry and Molecular Biology](https://bmb.natsci.msu.edu) at [Michigan State University](https://msu.edu))
- **Model type:** deep generative model
- **License:** Apache 2.0

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [https://github.com/giacomo-janson/sam2](https://github.com/giacomo-janson/sam2)
- **Paper:** [https://www.biorxiv.org/content/10.1101/2025.03.09.642148v1](https://www.biorxiv.org/content/10.1101/2025.03.09.642148v1)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

Direct uses include:
* Rapid generation of protein structural ensembles at atomic resolution.

* Exploration of protein conformational variability conditioned on temperature.

* Possible complement to molecular dynamics simulations in biomolecular research.

### Downstream Use

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->
Possible applications include:
* Functional predictions and interpretation of experimental data (e.g.: different thermostabilities in homologous proteins).

* Hybrid modeling combining ML-generated ensembles with physics-based methods (e.g.: provide starting conformations for MD runs, provide initial pools of conformations that can be reweighted with experimental data).

* Protein engineering and design (e.g.: evaluate the dynamical properties of different designs).

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

The model is intended primarily for research and experimental uses. Additional validation and development are required prior to deployment in practical, real-world applications.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Technical limitations, for the mdCATH-based aSAMt model:
* The model was trained on classical atomistic MD simulations of proteins, so it inherits the biases of this data (e.g.: force field inaccuracies). 
* Ensemble quality is dependent on training data representativeness:
    * Limited amount of sampling in training data. Each training system had at most 5 × 500 ns simulations at a given temperature, which for globular proteins is likely far from equilibrium.
    * Limited sequence diversity in the training data. In mdCATH, sequences are clustered at 20% sequence identity, so aSAMt may not capture differences in the ensembles of proteins with high sequence similarity.
* Potential inaccuracies when extrapolating to extreme temperatures outside the training range. We tested from 250 to 710 K.
* Potential inaccuracies when extrapolating to proteins with sequences significantly longer that the maximum length used in training. We tested the model up to 750 residue proteins.
* Trained on globular proteins, not expected to work well for intrinsically disordered proteins.
* Ensembles miss kinetically distant or rare states.
* Occasionaly generates structures with atomic clashes.
* Can be applied only to monomeric proteins.
* Can not model ligands, cofactors, post-translational modifications and disulfide bridges.

Sociotechnical limitations:
* The model is trained on MD data, but does not perfectly reproduce simulations. Therefore it should be considered as an empirical method, not as a substitute of MD simulations. As with any machine learning model, results should be critically evaluated and should not be trusted blindly.


### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

We recommend using this model only for applications similar to those described in the manuscript. Since the method is a data-driven machine learning model, we advice against applications on systems not represented in the training data.

## How to Get Started with the Model

Refer to the `README.md` file of the [repository of the model](https://github.com/giacomo-janson/sam2) to get started.

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

**ATLAS-based aSAMc**: we used the [ATLAS dataset](https://www.dsimb.inserm.fr/ATLAS/index.html), which consists of explicit solvent simulations of 1,390 proteins chains from the [Protein Data Bank](https://www.rcsb.org). For each chain, there are three 100 ns simulations. We utilized the same training, validation and test set splits of [AlphaFlow](https://github.com/bjing2016/alphaflow). Due to computational constraints, we removed all chains with more than 500 residues in the training and validation sets, resulting in 1,174 and 38 chains, respectively. For the test set, we used all 82 proteins, with lengths from 39 to 724.

**mdCATH-based aSAMt**: we used the [mdCATH dataset](https://github.com/compsciencelab/mdCATH), containing explicit solvent simulations of 5,398 protein domains from [CATH](https://www.cathdb.info). Simulations are available at five temperatures: 320, 348, 379, 413 and 450 K. For each domain and temperature, there are five trajectories of at most 500 ns. We randomly split the domains into 5,268 training, 40 validation and 90 test elements (the splits are available [here](https://github.com/giacomo-janson/sam2/tree/main/data/splits/mdcath)). Since mdCATH domains are already clustered at 20% sequence identity, this split effectively yields partitions that do not share highly similar domains. The maximum chain length of mdCATH is 500. For training the autoencoder and diffusion models, we applied a length cutoff of 320, resulting in 5,056 domains. We did not apply cutoffs for the validation and test sets.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

**Autoencoder**: the 3D coordinates of input protein conformations were converted in [OpenFold](https://github.com/aqlaboratory/openfold) representations (e.g.: backbone frames, side chain torsions) and saved as files loaded during training. This allowed faster scoring of the FAPE loss, avoiding the convertion step at training time.

**Latent diffusion model**: the 3D coordinates of input protein conformations were encoded using the encoder network of aSAM and saved as files loaded during training. This allowed faster training, avoiding the encoding step at training time.


#### Training Hyperparameters

- **Training regime:** fp16 mixed precision. For the number of training steps, batch size and other training details, please refer to the Supplementary Information in the [manuscript](https://www.biorxiv.org/content/10.1101/2025.03.09.642148v1).

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->
We evaluated model on the following tasks:

* Ability to replicate structural ensembles observed in MD simulations of proteins.
* Capturing experimentally observed melting temperatures of proteins.

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

**ATLAS-based aSAMc**: for the [ATLAS dataset](https://www.dsimb.inserm.fr/ATLAS/index.html), we utilized the same set split of [AlphaFlow](https://github.com/bjing2016/alphaflow), containing 82 proteins, with lengths from 39 to 724. We compared the model with the AlphaFlow and ESMFlow generative models trained on the same ATLAS data of aSAMc, as well as with baseline ensembles obtained with restrained [COCOMO2](https://github.com/feiglab/cocomo) coarse-grained simulations.

**mdCATH-based aSAMt**: for the [mdCATH dataset](https://github.com/compsciencelab/mdCATH), we used 90 test domains (the list is available [here](https://github.com/giacomo-janson/sam2/tree/main/data/splits/mdcath)). These domains have at most 20% sequence identity with any training set domain.

**mdCATH-based aSAMt, comparison with experimental $T_m$ values**: we used the experimentally measured $T_m$ values of 62 monomeric proteins from the a previously reported dataset used for the [SCooP method](https://academic.oup.com/bioinformatics/article/33/21/3415/3892394) for $T_m$ value prediction.


#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

**ATLAS-based aSAMc, comparison with MD**: we used the following series ensemble comparison and analysis scores (see the [manuscript](https://www.biorxiv.org/content/10.1101/2025.03.09.642148v1) for more information):
* Pearson correlation coefficient (PCC) of Cα root mean square fluctuation values (RMSF): we computed the Cα RMSF of the MD (reference) and generated ensembles and calculated the PCC between the two series.
* WASCO-global: "overall global discrepancy" as computed in the `comparison_tool.ipynb` notebook of [WASCO](https://gitlab.laas.fr/moma/methods/analysis/WASCO). Compares φ and psi torsion angle distributions in two ensembles.
* WASCO-local: "overall local discrepancy" as computed in the `comparison_tool.ipynb` notebook of [WASCO](https://gitlab.laas.fr/moma/methods/analysis/WASCO). Compares the torsion distribution of Cβ atoms in two ensembles.
* chiJSD: compares the distributions of side chain torsion angles χ in two ensembles.
* Heavy clashes: average number of heavy atom clashes in the snapshots of an ensemble.
* Peptide bond length violations: average number of peptide bond length deviations in the snapshots of an ensemble.

**mdCATH-based aSAMt, comparison with MD**: we used the following series ensemble analysis scores (see the [manuscript](https://www.biorxiv.org/content/10.1101/2025.03.09.642148v1) for more information):
* Average initRMSD of an ensemble: initRMSD is defined as the root mean squared deviation of a conformations in an ensemble with some reference structure, in this case the initial structure of an MD simulation.
* Average radius-of-gyration of an ensemble.
* Secondary structure element preservation (SSEP) in an ensemble.
* Folded state fraction (FSF) in an ensemble: the definition of folded state is based on the [fraction of native contacts](https://www.pnas.org/doi/10.1073/pnas.1311599110).

**mdCATH-based aSAMt, comparison with experimental $T_m$ values**:
* PCC between the experimental values and aSAMt estimations.

### Results 

**ATLAS-based aSAMc, comparison with MD**: aSAMc is slighly outperformed by AlphaFlow on PCC Cα RMSF and WASCO-global scores and aSAMc significantly outperforms AlphaFlow on WASCO-local and chiJSD (Table 1 in the [manuscript](https://www.biorxiv.org/content/10.1101/2025.03.09.642148v1)).

**mdCATH-based aSAMt, comparison with MD**: for each property, we count the number of test set domains ($n$=90) for which a aSAMt prediction matches within a threshold Δ the MD values across the 5 temperatures available in mdCATH (Supplementary Fig. 7 in the [manuscript](https://www.biorxiv.org/content/10.1101/2025.03.09.642148v1)).
* Average initRMSD values (Δ = 0.1 nm): 20/90 domains.
* Average radius-of-gyration (Δ = 0.1 nm): 60/90 domains.
* SSEP (Δ = 0.1): 51/90 domains.
* FSF: (Δ = 0.1): 23/90 domains.

**mdCATH-based aSAMt, comparison with experimental $T_m$ values**: aSAMt obtains a PCC of 0.510 ($p$=2.3e-5) with the experimental values (Fig. 7a in the [manuscript](https://www.biorxiv.org/content/10.1101/2025.03.09.642148v1)).


## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** RTX 2080 Ti
- **Hours used:** 2784 (for training the final autoencoder and diffusion model of aSAMt)
- **Cloud Provider:** academic infrastructure
- **Compute Region:** US
- **Carbon Emitted:** 300.67 kg CO2 eq.

## Technical Specifications

### Model Architecture and Objective

aSAM is latent diffusion model, trained in two stages. In the first, an autoencoder (AE) is trained to encode protein conformations into an SE(3)-invariant space. The reconstruction loss is mainly based on the FAPE loss used in [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2). In the second stage, a frozen encoder from the previous stage is used to encode protein conformations and their distribution is learned by a denoising diffusion probabilistic model (DDPM), utilizing the $L_{simple}$ objective introduced in the [original DDPM work](https://arxiv.org/abs/2006.11239).

### Compute Infrastructure

High performance computing systems at [MSU](https://msu.edu) and [ACCESS](https://allocations.access-ci.org).

#### Hardware

* Training GPU: 2-4 × NVIDIA RTX 2080 Ti
* Inference GPU: 1 × NVIDIA V100 32 GB, 1 × NVIDIA RTX 2080 Ti

#### Software

* Training and inference: PyTorch 1.13.1, PyTorch Lightning 2.1.3, Diffusers 0.30.0, 
* Evaluation data generation: OpemMM 8.0.0, MODELLER 10.4.
* Data analysis: custom Python 3.10 code, NumPy 1.21.5, Matplotlib 3.5.2, SciPy 1.7.3, Pandas 1.4.2, Seaborn 0.12.0, MDTraj 1.9.7, WASCO, PHENIX 1.20.1 (for computing MolProbity scores), PyEMMA 2.5.12, PyMOL 2.5.

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**
```
@article {Janson2025.03.09.642148,
	author = {Janson, Giacomo and Jussupow, Alexander and Feig, Michael},
	title = {Deep generative modeling of temperature-dependent structural ensembles of proteins},
	elocation-id = {2025.03.09.642148},
	year = {2025},
	doi = {10.1101/2025.03.09.642148},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/03/13/2025.03.09.642148},
	eprint = {https://www.biorxiv.org/content/early/2025/03/13/2025.03.09.642148.full.pdf},
	journal = {bioRxiv}
}
```

## Model Card Contact
* jansongi@msu.edu
* mfeiglab@gmail.com