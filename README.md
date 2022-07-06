 # Bayesian Modeling of Language-Evoked Event-Related Potentials
This repository contains the code for reproducing the results and images in "Bayesian Modeling of Language-Evoked Event-Related Potentials" (Accepted as a poster at [CCN 2022](https://2022.ccneuro.org/index.php)) and available as a preprint on arXiv.

### Setting up the environment and workspace
To create a conda environment (Python 3) with the required packages run:

`conda --name <myenv> --file requirements.txt`

Activate the environment:

`conda activate <myenv>`

Create  the following directories : `data/`, `images/`, `traces/` (e.g. `mkdir data`). Download the data from Frank et al (2015)[^fn] in `data/`.
[^fn]: Frank, S. L., Otten, L. J., Galli, G., & Vigliocco, G. (2015, 1). The ERP response to the amount of information conveyed by words in sentences. Brain and Language, 140, 1-11. doi: 10.1016/J.BANDL.2014.10.006

### Estimate surprisal values
In order to evaluate surprisal values for the LSTM, a PyTorch pre-trained model is needed; it should be called `model.pt` and saved in the main directory. The LSTM can be trained following [this PyTorch tutorial](https://github.com/pytorch/examples/tree/main/word_language_model). 
To estimate surprisal use:

```
python estimate_surprisal.py    --lm='ngram'
                                --cuda
``` 


with the following arguments:
| Parameter | Description  | Choices
|-----------------|:-------------:|---------------:|
| lm | Language model used for estimating surprisal  | 'ngram', 'lstm', 'gpt2'    |
| cuda     | Use CUDA (GPU) device if available |N/A |

Surprisal values from $n$-grams are already calculated for $n=3$ by Frank et al and saved in `surp.npy`.
### Fit the model and sample
For sampling the posterior use `python main.py --lm <model>` where `<model>` is the language model to use. For now, the sampler will sample 4 chains of 1000 draws (+1000 for warm-up): more flexibility will be allowed in a future release.

### Posterior predictive checks and figures 
Posterior predictive checks can be performed by using:
```
python post_checks.py    --lm='ngram'
                         --check='components'
```   
with the following parameters:

| Parameter | Description  | Choices
|-----------------|:-------------:|---------------:|
| lm | Language model used for estimating surprisal  | 'ngram', 'lstm', 'gpt2'    |
| check     | Posterior predictive check to perform |'contributions', 'tags', 'components', 'topography', 'all'|


The script will save associated figures in the `images/` folder. Diagnostics plots can be obtained by using the Jupyter notebook `diagnostics_plots.ipynb`.
