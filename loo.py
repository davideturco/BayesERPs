import arviz as az
import pandas as pd
import numpy as np
import pickle as pkl
from models import HierarchicalModel_complete_nc_type


def loo(trace):
    """Calculate pointwise LOO elpd for given trace."""
    loos = az.loo(trace, pointwise=True)
    return np.array(loos.loo_i)


def main():
    trace_ngram = az.from_netcdf('../scratch/test3.nc')
    trace_lstm = az.from_netcdf('../scratch/trace_lstm2.nc')
    trace_gpt2 = az.from_netcdf('../scratch/trace_gpt2_2.nc')
    df_ngram = pd.read_csv('df_ngram.csv')
    df_lstm = pd.read_csv('df_lstm.csv')
    df_gpt2 = pd.read_csv('df_gpt2.csv')
    model_ngram = HierarchicalModel_complete_nc_type(df_ngram)
    model_lstm = HierarchicalModel_complete_nc_type(df_lstm)
    model_gpt2 = HierarchicalModel_complete_nc_type(df_gpt2)

    compare_dict = {'model_ngram': trace_ngram, 'model_lstm': trace_lstm,
                    'model_gpt2': trace_gpt2}
    with open('loos.pkl', 'wb') as f:
        comp = az.compare(compare_dict)
        pkl.dump(comp, f)

    np.save('loo_ngram.npy', loo(trace_ngram))
    np.save('loo_lstm.npy', loo(trace_lstm))
    np.save('loo_gpt2.npy', loo(trace_gpt2))


if __name__ == '__main__':
    main()
