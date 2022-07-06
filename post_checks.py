import arviz as az
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as lines
import matplotlib.animation as animation
from itertools import combinations
import pickle as pkl
from models import HierarchicalModel_complete_nc_type, LinearModel
from utils import plot_contribs
from tag_sentences import get_tags
import argparse
from utils import get_erps, chan_list, create_info_obj, plot_topography

plt.style.use('ggplot')
matplotlib.rcParams.update({'font.size': 11})
content = ['ADV', 'ADJ', 'NOUN', 'VERB']
non_content = ['NUM', 'DET', 'PRON', 'ADP', 'PRT', 'CONJ']


def estimate_contrib(trace, item):
    df = pd.DataFrame(az.summary(trace, var_names=item))
    ctr = np.array(df.loc[[i for i in df.index.values if i.startswith(f'{item}[1')]]['mean']).mean()
    return ctr


def categorise(row):
    if row in content:
        cat = 'CONT'
    else:
        cat = 'FUNC'
    return cat


def check_convergence(trace):
    """Check convergence in terms of R-hat statistic and effective sample size (ess).
    :param trace: PyMC3 trace
    :return: upper bound for R-hat (float), lower bound for ess percentange (float)

    """
    num_samples = len(trace.posterior.draw)
    summary = pd.DataFrame(az.summary(trace, round_to=4))
    rhat = summary.r_hat.max()
    ess_percentage = (summary.ess_bulk.min() / num_samples) * 100
    print(f'Upper_bound for r_hat: {rhat}, lower bound for ess percentage: {ess_percentage}%')

    return rhat, ess_percentage


def post_predictive(data, samples, trace, model):
    """
    Plot posterior predictive samples showing both the posterior approximation and the fit to the observed data
    :param data: Pandas dataframe
    :param samples: Posterior predictive samples as calculated by pymc3.sample_posterior_predictive()
    :param trace: PyMC3 trace
    :param model: PyMC3 model
    """
    _, axs = plt.subplots(2, 1)
    # posterior predictive check
    az.plot_ppc(az.from_pymc3(posterior_predictive=samples, model=model.nc), num_pp_samples=1000, ax=axs[0])
    # show fit of data
    mu_pred = np.asarray(trace.posterior['mu'])
    axs[1].scatter(data.surprisal, data.ERP, c='C0', alpha=0.4, facecolors="none")
    az.plot_hdi(data.surprisal, samples['erp'], ax=axs[1])
    az.plot_hdi(data.surprisal, mu_pred, fill_kwargs={'alpha': 0.8}, ax=axs[1])
    axs[1].set_xlabel("surprisal (a.u.)")
    axs[1].set_ylabel("N400 (a.u.)")
    # plt.show()
    plt.savefig('images/ppc.png')


def estimate_contribs(data, trace):
    """
    Estimate and plot the by-subject contributions to surprisal for a traditional linear model and a given hierarchical
    model
    :param data: Pandas Dataframe
    :param trace: PyMC3 trace
    """
    b = trace.posterior.b.mean().item()
    b_sub = estimate_contrib(trace.posterior, 'ab_sub')
    b_t = estimate_contrib(trace.posterior, 'ab_t')
    try:
        b_e = estimate_contrib(trace.posterior, 'ab_e')
    except KeyError:
        b_e = 0

    print(f'Individual contribs.: {b, b_sub, b_t, b_e}')
    print(f'Overall surprisal effect: {b + b_sub + b_t + b_e}')

    print(az.summary(trace, var_names='ab_sub', round_to=3))

    # forest plots for subject contribution
    # fit a traditional linear model first
    lin_model = LinearModel(data)
    coefs = lin_model.fit()
    print(f'Overall contribution for the frequentist model: {coefs.mean()}')

    hier = np.array(
        trace.posterior.ab_sub[:, :, 1, :].mean(axis=(0, 1))) + trace.posterior.b.mean().item() + trace.posterior.ab_t[
                                                                                                  :, :, 1,
                                                                                                  :].mean().item() + trace.posterior.ab_e[
                                                                                                                     :,
                                                                                                                     :,
                                                                                                                     1,
                                                                                                                     :].mean().item()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot([coefs, hier], [np.arange(24)[::-1], np.arange(24)[::-1]], "k-", alpha=0.3)
    ax.axvspan(coefs.mean() - coefs.std(), coefs.mean() + coefs.std(), alpha=0.5, zorder=1, color='#F8766D')
    ax.scatter(coefs, np.arange(24)[::-1], label='Unpooled freq.', marker='D')
    ax.axvspan(coefs.mean() - coefs.std(), coefs.mean() + coefs.std(), alpha=0.2, zorder=1, color='#F8766D')
    ax.scatter(hier, np.arange(24)[::-1], label='Hierarch. Bayes.')
    ax.axvspan(hier.mean() - hier.std(), hier.mean() + hier.std(), alpha=0.2, zorder=1, color='#00BFC4')
    ax.set_yticks(np.arange(0, 24))
    ax.set_yticklabels(np.arange(1, 24 + 1)[::-1])
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax.set_ylabel('Subject', color='k')
    ax.set_xlabel('Posterior effect', color='k')
    ax.tick_params(axis='both', colors='k')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'images/{lm}/shrinkage.pdf')


def word_contrib(trace):
    """
    Estimate and plot POS tag and content/function words contributions to surprisal
    :param trace: PyMC3 trace
    """
    print('Computing POS tags...')
    tags = get_tags()
    print('Done')

    try:
        # Group contribution by tag
        summary = pd.DataFrame(az.summary(trace.posterior, var_names='ab_i'))
        summary = summary.loc[[i for i in summary.index.values if i.startswith(f'ab_i[1')]]
        summary['tag'] = tags
        ranked = summary.groupby('tag').mean().sort_values(by='mean')
        print(ranked)

        _, (ax1, ax2) = plt.subplots(1, 2)
        plot_contribs(ranked, ax=ax1)

        # Group tags by function/content words
        ranked.reset_index(level=0, inplace=True)
        ranked['categ'] = ranked['tag'].apply(categorise)
        cats = ranked.groupby('categ').mean()
        print(f'\n{cats}')

        plot_contribs(cats, ax=ax2)
        plt.show()

    except KeyError:
        summary = pd.DataFrame(az.summary(trace.posterior, var_names='ab_t', round_to=3))
        print(summary)

        fig, axs = plt.subplots(2, 1, figsize=(5, 3.5))
        az.plot_forest(trace, var_names='ab_t', coords={'ab_t_dim_0': 1}, combined=True, kind='ridgeplot',
                       ridgeplot_alpha=.3, ax=axs[0], hdi_prob=.97, colors='g')
        axs[0].set_yticks(np.arange(2))
        axs[0].set_yticklabels(['FUNC', 'CONT'], color='k')
        axs[0].set_xlabel(r'Posterior effect', color='k')
        axs[0].tick_params(axis='both', colors='k')
        # plot posterior distr. of the difference to investigate "significance"
        diff = trace.posterior.ab_t[:, :, 1, 1] - trace.posterior.ab_t[:, :, 1, 0]
        diff = diff.data.reshape(-1)
        pd.DataFrame(diff).plot.kde(ax=axs[1], legend=False, c='g')
        axs[1].set_xlabel(r'Posterior difference (FUNC-CONT)', color='k')
        axs[1].set_ylabel(r'Density', color='k')
        axs[1].tick_params(axis='both', colors='k')
        plt.tight_layout()
        plt.savefig(f'images/{lm}/tags.pdf')


def erp_contrib(trace):
    """
    Estimate and plot the contributions of the six ERP components
    :param trace: PyMC3 trace
    """
    comp_names = ["ELAN", "LAN", "N400", "EPNP", "P600", "PNP"]
    _, axs = plt.subplots(1, 2, figsize=(13, 4))

    # Plot correlations of the samples from the different ERP components
    # subs = trace.posterior.ab_e.mean(axis=0).values[:, 1, :]
    # subs = trace.posterior.ab_e.values[:, :, 1, :].reshape(-1, trace.posterior.ab_e.shape[-1])
    # to_corr = pd.DataFrame(subs)
    # corrs = to_corr.corr(method='spearman')
    # print(corrs)
    # mask = np.zeros_like(corrs)
    # mask[np.triu_indices_from(mask)] = True
    # sns.heatmap(corrs, cmap='PuBu', mask=mask, annot=True, cbar_kws={'label': 'Spearman rank corr.'},
    #             ax=axs[0], yticklabels=comp_names, xticklabels=comp_names, square=True)

    # _, axs = plt.subplots(1, 2, figsize=(5, 3.5))
    # summary = az.summary(trace, var_names=['ab_e'], round_to=3)
    # summary = summary.loc[[i for i in summary.index.values if i.startswith(f'ab_e[1')]]
    # print(summary)
    # plot_contribs(summary, ax=axs[0], color='#9400D3')
    # axs[0].set_yticks(np.arange(6))
    # axs[0].set_yticklabels(comp_names, color='k')
    # axs[0].set_xlabel('Posterior effect', color='k')
    # axs[0].tick_params(axis='both', colors='k')
    # # Plot posterior differences
    # ERPS = dict(zip(np.arange(6), ["ELAN", "LAN", "N400", "EPNP", "P600", "PNP"]))
    # combs = list(combinations(np.arange(6), r=2))
    # diffs = np.zeros((len(combs), 3))
    # for i, comb in enumerate(combs):
    #     diff = trace.posterior.ab_e[:, :, 1, comb[0]] - trace.posterior.ab_e[:, :, 1, comb[1]]
    #     hdis = az.hdi(diff, hdi_prob=.97)
    #     diffs[i] = diff.mean().item(), hdis.ab_e.values[0], hdis.ab_e.values[1]
    # labels = [f'{ERPS[combs[i][0]]} - {ERPS[combs[i][1]]}' for i in range(len(combs))]
    # summary = pd.DataFrame(data=diffs[:, [0, 2]], index=np.arange(len(combs)), columns=['mean', 'hdi_97%'])
    # summary.sort_values(by=['mean'], ascending=False, inplace=True)
    # plot_contribs(summary, ax=axs[1], color='#9400D3')
    # ordered_labels = [labels[j] for j in summary.index.values]
    # axs[1].set_yticklabels(ordered_labels)
    # axs[1].set_xlabel('Posterior difference', color='k')
    # axs[1].tick_params(axis='both', colors='k')
    # blue_patch = lines.Line2D([], [], color='#9400D3', marker='_', linestyle='None',
    #                           markersize=10, markeredgewidth=1.5, label=f'97%\nHDI')
    # axs[1].legend(handles=[blue_patch])
    # plt.tight_layout()
    # plt.savefig('images/components.png')

    _, ax = plt.subplots(figsize=(5, 3.5))
    summary = az.summary(trace, var_names=['ab_e'], round_to=3)
    summary = summary.loc[[i for i in summary.index.values if i.startswith(f'ab_e[1')]]
    print(summary)
    plot_contribs(summary, ax=ax, color='#9400D3')
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(comp_names, color='k')
    ax.set_xlabel('Posterior effect', color='k')
    ax.tick_params(axis='both', colors='k')
    blue_patch = lines.Line2D([], [], color='#9400D3', marker='_', linestyle='None',
                              markersize=10, markeredgewidth=1.5, label=f'97%\nHDI')
    ax.legend(handles=[blue_patch], loc='upper left')
    plt.tight_layout()
    plt.savefig(f'images/{lm}/components.pdf')


def topography(trace):
    """
    Plots topographic map of contributions of each electrode (based on the assumptions about electrode involvement in
    ERPs for a given time window). Results are plotted both as an animated gif which shows the topography for different
    time windows, and as a static plot showing the averaged electrode contribution.
    :param trace: PyMC3 trace
    :return:
    """
    comps = get_erps()
    chans = chan_list(comps)
    contribs = trace.posterior.ab_e[:, :, 1, :].mean(dim=['chain', 'draw']).values
    conds = [np.arange(125, 176), np.arange(300, 400), np.arange(400, 500), np.arange(500, 600), np.arange(600, 700)]
    info = create_info_obj(chans)

    values = np.zeros((len(conds), len(chans)))
    for c, cond in enumerate(conds):
        for j, comp in enumerate(comps):
            if set(comp.time_window) & set(cond):
                for i, chan in enumerate(chans):
                    if chan in comp.channels:
                        values[c, i] = values[c, i] + contribs[j]
                    # else:
                    #     values[c, i] = values[c, i]
    # average across the 6 ERP components
    values = values / 6
    fig, ax = plt.subplots()
    labels = ['<300 ms', '300-400 ms', '400-500 ms', '500-600 ms', '600-700 ms']
    vmin = values.min()
    vmax = values.max()

    # utils funct for animation
    def animate(k):
        data = values[k, :]
        plot_topography(data, chans=chans, lims=(vmin, vmax), info=info, ax=ax, cbar=True)
        ax.set_title(f'{labels[k]}')

    anim = animation.FuncAnimation(fig, animate, frames=5, interval=1000, repeat=True)
    f = rf"images/{lm}/anim.gif"
    writergif = animation.PillowWriter(fps=2)
    anim.save(f, writer=writergif)

    # plot also a static image averaged across all time windows
    avgd = np.mean(values, axis=0)
    fig1, ax1 = plt.subplots()
    plot_topography(avgd, chans=chans, info=info, cbar=True, ax=ax1)
    plt.savefig(f'images/{lm}/topography_averaged.pdf')


def main(option):
    df = pd.read_csv(f'df_{lm}.csv')
    # if not np.array_equal(df.Participant.unique(), np.arange(12)):
    #     df.Participant = df.Participant.replace(dict(zip(df["Participant"].unique(), np.arange(12))))
    model = HierarchicalModel_complete_nc_type(df)
    trace = az.from_netcdf(f'traces/trace_{lm}.nc')
    # with open(ppc_path, 'rb') as f:
    #     ppcs = pkl.load(f)

    if option == 'ppc':
        post_predictive(df, ppcs, trace, model)
    elif option == 'contributions':
        estimate_contribs(df, trace)
    elif option == 'tags':
        word_contrib(trace)
    elif option == 'components':
        erp_contrib(trace)
    elif option == 'topography':
        topography(trace)
    elif option == 'all':
        estimate_contribs(df, trace)
        word_contrib(trace)
        erp_contrib(trace)
        topography(trace)
    check_convergence(trace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Posterior predictive checks')

    # parser.add_argument('-ppc', type=str, default='ppc_nc.pkl', help='Path to the PPC samples (str.)')
    parser.add_argument('-lm', choices=['ngram', 'lstm', 'gpt2'],
                        help='Language model used [\'ngram\', \'lstm\', \'gpt2\']')
    parser.add_argument('-check', choices=['ppc', 'contributions', 'tags', 'components', 'topography', 'all'],
                        help='Post-predictive check to be performed.')
    args = parser.parse_args()
    lm = args.lm
    main(args.check)
