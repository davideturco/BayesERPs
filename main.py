from models import HierarchicalModel_complete_nc_type
import pandas as pd
import warnings
import argparse

warnings.filterwarnings("ignore")


def main(lm):
    data = pd.read_csv(f'df_{lm}.csv')[['Participant', 'component', 'tags', 'surprisal', 'ERP']]
    data.tags = pd.Categorical(data.tags)
    data['tags'] = data.tags.cat.codes
    data.component = pd.Categorical(data.component)
    data['component'] = data.component.cat.codes

    # if not np.array_equal(data.Participant.unique(), np.arange(12)):
    #     data.Participant = data.Participant.replace(dict(zip(data["Participant"].unique(), np.arange(12))))
    model = HierarchicalModel_complete_nc_type(data)
    with model.nc:
        trace = model.sample(num=1000, tune=1000, cores=1, target=.95)
        trace.to_netcdf(f'traces/trace_{lm}.nc')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample using NUTS HMC. Please, specify dataset to fit model on')
    parser.add_argument('-lm', choices=['ngram', 'lstm', 'gpt2'],
                        help='Language model used for calculating surprisal values on which fit the model [\'ngram\', '
                             '\'lstm\', \'gpt2\']')
    args = parser.parse_args()

    main(args.lm)
