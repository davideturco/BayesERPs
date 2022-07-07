 # Bayesian Modeling of Language-Evoked Event-Related Potentials
This repository contains the code for reproducing the results and images in "Bayesian Modeling of Language-Evoked Event-Related Potentials" (Accepted as a poster at [CCN 2022](https://2022.ccneuro.org/index.php)) and available as a preprint on arXiv.

## Setting up the environment and workspace
To create a conda environment (Python>=3.9) with the required packages run:

`conda env create --file envname.yml`

Activate the environment:

`conda activate <envname>`


<!---
If faster sampling with JAX is required, install the appropriate package using:

`pip install numpyro`

-->

Create  the following directories : `data/`, `images/`, `traces/` (e.g. `mkdir data`). Download the data from Frank et al (2015)[^fn] in `data/`. The `images/` should have three directories corresponding to the three language models: `ngram/`, `lstm/`, `gpt2/`.
[^fn]: Frank, S. L., Otten, L. J., Galli, G., & Vigliocco, G. (2015, 1). The ERP response to the amount of information conveyed by words in sentences. Brain and Language, 140, 1-11. doi: 10.1016/J.BANDL.2014.10.006

## Estimate surprisal values
In order to evaluate surprisal values for the LSTM, a PyTorch pre-trained model is needed.  The user should checkout [this](https://github.com/pytorch/examples/tree/main/word_language_model) PyTorch repository in a new directory called `lstm/` and follow the tutorial to train the model. 
Then, to estimate surprisal use:

```
python estimate_surprisal.py    --lm=ngram
                                --cuda
``` 


with the following arguments:

| Parameter | Description  | Choices
|-----------------|:-------------:|---------------:|
| lm | Language model used for estimating surprisal  | 'ngram', 'lstm', 'gpt2'    |
| cuda     | Use CUDA (GPU) device if available |N/A |

Surprisal values from $n$-grams are already calculated for $n=3$ by Frank et al and saved in `surp.npy`.

## Generate the dataset
Once the surprisal values are saved, a dataset can be generated with:
```
python data_processing.py    --erp=all
                             --lm-ngram
                             --sj=24
                             --nw=1931
                             --wd=type
                             --sv=True
```
with the following parameters:

| Parameter | Description  | Choices
|-----------------|:-------------:|---------------:|
| erp | ERP components to include  | 'all', 'N400'|
| lm     | Language model used for estimating surprisal  | 'ngram', 'lstm', 'gpt2'    |
|sj | Number of subjects  | default=24 (all)    |
|nw | Number of words  | default=1931 (all)    |
|wd | Include all words or only word type/tags  | 'words', 'type'|
|sv | Save the dataset as csv file  |True, False|
## Fit the model and sample
For sampling the posterior use
```
python main.py    --lm=ngram
                  --backend=default
```

| Parameter | Description  | Choices|
|-----------------|:-------------:|---------------:|
| lm | Language model used for estimating surprisal  | 'ngram', 'lstm', 'gpt2'    |
| backend     | Back-end to use for sampling. Sampling with Jax requires a GPU. |'default', 'jax'|

For now, the sampler will sample 4 chains of 1000 draws (+1000 for warm-up): more flexibility will be allowed in a future release.

## Posterior predictive checks and figures 
Posterior predictive checks can be performed by using:
```
python post_checks.py    --lm=ngram
                         --check=components
```   
with the following parameters:

| Parameter | Description  | Choices
|-----------------|:-------------:|---------------:|
| lm | Language model used for estimating surprisal  | 'ngram', 'lstm', 'gpt2'    |
| check     | Posterior predictive check to perform |'contributions', 'tags', 'components', 'topography', 'all'|


The script will save associated figures in the `images/` folder. Diagnostics plots can be obtained by using the Jupyter notebook `diagnostics_plots.ipynb`.