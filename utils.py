import numpy as np
from scipy.io import loadmat
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pymc as pm
import arviz as az
import mne


#######################
## Data processing
#######################

def array_from_mat(file, key, sentence):
    array = file[key]
    return np.concatenate(array[sentence], axis=0)


def get_sentences(file, key, num_sentences):
    lst_sent = [array_from_mat(file, key, s) for s in np.arange(num_sentences)]
    return lst_sent


def get_features(file, feature, num_sentences):
    raw_feat = get_sentences(file, feature, num_sentences)
    if raw_feat[0].shape[0] == 1:
        raw_feat = [i.T for i in raw_feat]
    return np.vstack(raw_feat).flatten()


def check_missing(df):
    num = df.isnull().sum()
    subjects = df[df.isnull().any(axis=1)].Participant.unique()
    print(f'Missing values:\n{num}.\n\nParticipants with missing values are {subjects}.')


def drop_missing(df):
    return df.dropna(inplace=True)


def normalise(arr):
    """
    Normalise an array to the range [0,1]
    :param arr: NumPy array to normalise
    :return: Normalised NumPy array
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def correlation(arr1, arr2):
    """
    Calculate Pearson correlation coefficient between two arrays
    :param arr1: NumPy array
    :param arr2: NumPy array
    :return: Pearson correlation coeff (float)
    """
    return np.corrcoef(arr1, arr2)[0][1]


#######################
## Plotting
#######################

def Gauss2d(mu, cov, ci, ax=None):
    """Copied from statsmodel and Statistical Rethinking. Creates contours of confidence for a distribution
    with mean mu and covariance matrix cov"""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    v_, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees

    for level in ci:
        v = 2 * np.sqrt(v_ * stats.chi2.ppf(level, 2))  # get size corresponding to level
        ell = Ellipse(
            mu[:2],
            v[0],
            v[1],
            180 + angle,
            facecolor="None",
            edgecolor="k",
            alpha=(1 - level) * 0.5,
            lw=1.5,
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

    return ax


def ppc(trace, model, var='erp', ax=None):
    """
    Plots posterior predictive checks.
    Not suggested for heavy traces (lots or parameters or samples)
    :param trace: trace of the model in Pickle format
    :param model: PyMC3 model
    :param num_samples: number of samples from the posterior
    :param var: variable to check
    :param ax: Matplotlib axis object
    :return: Matplotlib axis object
    """
    if ax is None:
        _, ax = plt.subplots()

    pred = pm.sample_posterior_predictive(trace, model=model, var_names=[var])
    az.plot_ppc(az.from_pymc3(posterior_predictive=pred, model=model), ax=ax)
    return ax


def forest_plot(data, hdi=0.97, ax=None):
    """
    Given a PyMC3 trace or a NumPy array, it plots a forest plot with specified highest density interval (hdi)
    :param data: PyMC3 trace (also in Pickle format) or Numpy array
    :param hdi: Highest density interval
    :param ax: Matplotlib axis object
    :return: Matplotlib axis object
    """
    if ax is None:
        ax = plt.gca()
    if isinstance(data, np.ndarray):
        ax.scatter(x=data, y=(np.arange(len(data)) + 1)[::-1], facecolors='white', edgecolors='tab:blue', alpha=1,
                   zorder=2)
    else:
        try:
            ints = az.hdi(data, var_names='ab_sub', hdi_prob=hdi).ab_sub[1, :, :].values
            ints = [i[1] for i in ints]
            # ints = mean_hdi(ints)
            data = data.ab_sub.mean(axis=(0, 1))[1, :].values
        except KeyError:
            ints = az.hdi(data, var_names='b', hdi_prob=hdi).b.values
            ints = mean_hdi(ints)
            data = data.b.mean(axis=(0, 1)).values
        ax.errorbar(x=data, y=(np.arange(len(data)) + 1)[::-1], xerr=np.abs(ints - data), fmt='o', mec='tab:blue',
                    mfc='w', zorder=2)
    std = data.std()
    ax.set_yticks(np.arange(1, len(data) + 1))
    ax.set_yticklabels(np.arange(1, len(data) + 1)[::-1])
    ax.set_ylabel('Subject')
    # ax.set_xlabel('Surprisal effect (A.U.)')
    ax.axvline(x=data.mean(), c='orange')
    ax.axvspan(data.mean() - std, data.mean() + std, alpha=0.2, color='orange', zorder=1)
    return ax


def plot_contribs(summary, color='tab:blue', ax=None):
    """Given a Pandas summary in arviz format, plots contributions to surprisal as a forest plot
    :param summary: Summary in PyMC3 format (Pandas dataframe)
    :return: Axis object"""
    if ax is None:
        ax = plt.gca()

    ax.errorbar(x=summary['mean'].values, y=np.arange(len(summary.index.values)),
                xerr=np.abs(summary['mean'].values - summary['hdi_97%']), fmt='o',
                mec=color, c=color, mfc='w', zorder=2)
    ax.set_yticks(np.arange(len(summary)))
    ax.set_yticklabels(summary.index.values)
    ax.axvline(x=0, ls='--', lw=1, c='#e25822')
    return ax


def plot_topography(values, chans, info, lims=(None, None), cbar=False, ax=None):
    """
    Plots a static topographic map using MNE functions
    :param values: values to be plotted (NChans,)
    :param chans: names of the channels (NChans,)
    :param info: MNE info object
    :param cbar: Whether to include the color bar or not (Bool.)
    :param ax: Matplotlib axis object
    :return: Matplotlib axis object
    """
    if ax is None:
        ax = plt.gca()

    fig = plt.gcf()
    im, cm = mne.viz.plot_topomap(data=np.abs(values), show=False, vmin=lims[0], vmax=lims[1], names=chans, pos=info,
                                  cmap='RdBu_r', show_names=True, axes=ax)
    ax_x_start = 0.85
    ax_x_width = 0.04
    ax_y_start = 0.1
    ax_y_height = 0.8
    if cbar:
        cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im, cax=cbar_ax)
        clb.ax.set_title('arb. unit', fontsize=10)
    return ax


#######################
## ERPs processing
#######################
class ERPComponent:
    def __init__(self, component, channels, time):
        self.component = component
        self.channels = channels
        self.file = loadmat('data/stimuli_erp.mat')
        self.time_window = np.arange(time[0], time[1] + 1)


def get_erps():
    elan = ERPComponent('ELAN', [8, 21, 22, 33, 34, 37, 49], (125, 175))
    lan = ERPComponent('LAN', [8, 18, 33, 34, 48, 49], (300, 400))
    n400 = ERPComponent('N400', [1, 14, 24, 25, 26, 29, 30, 31, 41, 42, 44, 45], (300, 500))
    epnp = ERPComponent('EPNP', [35, 36, 37, 49, 50], (400, 600))
    p600 = ERPComponent('P600', [1, 12, 14, 16, 24, 25, 26, 29, 30, 31, 39, 40, 41, 42, 44, 45, 46, 47], (500, 700))
    pnp = ERPComponent('PNP', [1, 8, 10, 18, 21, 22, 24, 31, 33, 34], (600, 700))
    comps = [elan, lan, n400, epnp, p600, pnp]
    return comps


def chan_list(components):
    """
    Return a list of the channels used in the EEG experiment
    :param components: list of ERP-component objects
    :return: set of channels (set)
    """
    chans = []
    for comp in components:
        chans += comp.channels
    chans = set(chans)

    # Add electrode 38 which is not involved by any ERP component
    chans.add(38)

    return chans


def create_info_obj(channels):
    """
    Create a MNE-style info object
    :param channels: list or set of channels involved
    :return: MNE info object
    """
    info = mne.create_info(
        ch_names=[str(x) for x in list(channels)],
        ch_types=['eeg'] * 32,
        sfreq=500
    )

    info.set_montage('easycap-M10')
    return info


#######################
## Miscellaneous
#######################

def mean_hdi(itvs):
    """
    Convert an array of lower and upper highest density intervals (itvs) into errors that can be plotted
    :param itvs: lower(s) and upper(s) values of the interval(s) (NumPy array)
    :return: NumPy array
    """
    for i in range(itvs.shape[0]):
        itvs[i, :] = np.abs(itvs[i, :] - itvs[i, :].mean())
    return itvs
