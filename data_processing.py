import numpy as np
import pandas as pd
from utils import *
import argparse
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tag_sentences import get_tags, load_sentences
from post_checks import categorise
import estimate_surprisal

NUM_SENTENCES = 205
NUM_PARTICIPANTS = 24
NUM_WORDS = 1931
ERPS = dict(zip(["ELAN", "LAN", "N400", "EPNP", "P600", "PNP"], np.arange(6)))
LMS = dict(zip(['ngram', 'rnn'], ['surp_ngram', 'surp_rnn']))


def find_sent_limits(sentences):
    """
    Find indices of where each sentence starts
    :param sentences: lists of sentences (list of lists of strings)
    :return: lists of indices (list)
    """
    init = 0
    sent_delimiter = [0]
    for i in sentences[:-1]:
        init += len(i)
        sent_delimiter.append(init)
    return sent_delimiter


def filter_words(sentences, words):
    """
    Filters out from a list of word indices the following words: initial/final words, clitics, words attached to a comma
    or other punctuations.
    :param sentences: lists of sentences (list of lists of strings)
    :param words: list of words (list)
    :return: list of indices of words after filtering (list)
    """
    sent_delimiter = find_sent_limits(sentences)
    to_remove = []
    for j, w in enumerate(words):
        # TEST: just remove sentence initials
        # if any(c for c in w if c not in string.ascii_letters) or (j in sent_delimiter) or (
        #         j in (np.array(sent_delimiter) - 1)[1:]):
        if j in sent_delimiter:
            print(w)
            to_remove.append(j)

    idx = list(np.arange(NUM_WORDS))
    for index in sorted(to_remove, reverse=True):#[1:]:
        del idx[index]

    return idx


def assign_words(sentences):
    """
    For a given sequence of word indexes, this function returns a dictionary assigning each word to the index of the
    sentence it belongs to.
    :param sentences: lists of sentences (list of lists of strings)
    :return: dict
    """
    # init = 0
    # sent_delimiter = [0]
    # for i in file['sentences'].flatten()[:-1]:
    #     init += i.flatten().shape[0]
    #     sent_delimiter.append(init)
    sent_delimiter = find_sent_limits(sentences)
    test_words = np.arange(NUM_WORDS)
    words = []
    for i in test_words:
        for j, w in enumerate(sent_delimiter):
            try:
                if sent_delimiter[j] <= i < sent_delimiter[j + 1]:
                    words.append(j)
            except IndexError:
                words.append(j)

    word_pos = dict(zip(np.arange(NUM_WORDS), words))
    return word_pos


def deal_with_missing(df):
    """
    Checks for missing data and removes them.
    :param df: Pandas dataframe
    :return: Pandas dataframe
    """
    print('For the original dataframe:')
    print(check_missing(df))
    print('\nRemoving missing values...\n')
    drop_missing(df)


def select_subjects(df, subjects):
    df = df.loc[df['Participant'].isin(subjects)]
    return df


def select_words(df, num_words):
    words = np.arange(num_words)
    df = df.loc[df['word'].isin(words)]
    return df


def standardize_features(df, features):
    """
    Standardise covariates of interest.
    :param df: Pandas dataframe
    :param features: list of covariates to standardise
    :return: Pandas dataframe
    """
    print('Standardising covariates...')
    df[features] = StandardScaler().fit_transform(df[features])
    return df


def get_surprisal(file, sentences, lm):
    if lm == 'ngram':
        surps = get_sentences(file, LMS[lm], NUM_SENTENCES)
        surps = np.vstack(surps)[:, 2]
    elif lm == 'lstm':
        surps, _ = estimate_surprisal.estimate_surp_lstm(sentences)
    elif lm == 'gpt2':
        surps, _ = estimate_surprisal.estimate_surp_gpt2(sentences, incl=False)
    return surps


def create_interaction(df):
    """
    Create 2nd-order interaction terms for given dataframe.
    :param df: Pandas dataframe
    :return: Pandas dataframe with interaction terms
    """
    x = df.drop(['ERP', 'word', 'Participant'], axis=1)
    poly = PolynomialFeatures(interaction_only=True, degree=2)
    new = pd.DataFrame(poly.fit_transform(x, df.ERP))
    new = new.rename(columns=dict(zip(np.arange(1, 6), x.columns))).iloc[:, 1:]
    # new.index = df.index
    new['ERP'], new['word'], new['Participant'] = df['ERP'].values, df['word'].values, df['Participant'].values
    return new


def generate_dataframe(file, ERP_component, lan_model, subjects, num_words, word_choice, inter=False):
    """
    Given ERP data and features of interest extracted from a MATLAB file, generate a processed Pandas dataframe.
    :param file: a MATLAB .mat file loaded with SciPy's loadmat function
    :param ERP_component: str. ERP component to extract data for (see ERPS dict on top of this module)
    :param lan_model: str. Language model to extract data for
    :param subjects: int. Number of subjects to consider
    :return: Pandas dataframe
    """
    # load sentences and words
    sentences = load_sentences(file)
    words = [w for sent in sentences for w in sent]

    # load variables of interest
    ordr = file['sentence_position'] - 1
    # surps = get_sentences(file, LMS[lan_model], NUM_SENTENCES)
    # surps = np.vstack(surps)[:, 2]
    surps = get_surprisal(file, sentences, lan_model)
    logwordfreq = get_features(file, 'logwordfreq', NUM_SENTENCES)
    wordlen = get_features(file, 'wordlength', NUM_SENTENCES)
    erps = get_sentences(file, 'ERP', NUM_SENTENCES)

    if ERP_component == 'N400':
        erps_comp = np.vstack(erps)[:, :, ERPS[ERP_component]]
    elif ERP_component == 'all':
        erps_comp = np.vstack(erps)
        erps_comp = np.moveaxis(erps_comp, 2, 1)
        erps_comp = erps_comp.reshape(-1, NUM_PARTICIPANTS, order='F')
    else:
        raise NameError

    df_test = pd.DataFrame(erps_comp, columns=[np.arange(NUM_PARTICIPANTS)])
    df = pd.melt(df_test, ignore_index=False, value_vars=[np.arange(NUM_PARTICIPANTS)], var_name='Participant',
                 value_name='ERP')

    if ERP_component == 'all':
        df['component'] = np.tile(np.repeat(np.arange(6), repeats=NUM_WORDS), NUM_PARTICIPANTS)

    word_pos = assign_words(sentences)

    df['word'] = np.tile(np.arange(NUM_WORDS), len(df) // NUM_WORDS)
    df['surprisal'] = df.apply(lambda row: surps[int(row.word)], axis=1)
    df['orig_sent_pos'] = df.apply(lambda row: word_pos[int(row.word)], axis=1)
    df['sent_pos'] = df.apply(lambda row: int(ordr[int(row.Participant) - 1, int(row.orig_sent_pos)]), axis=1)
    df['logwordfreq'] = df.apply(lambda row: logwordfreq[int(row.word)], axis=1)
    df['wordlen'] = df.apply(lambda row: wordlen[int(row.word)], axis=1)
    df['wordpos'] = df.groupby('orig_sent_pos').cumcount()
    df.drop('orig_sent_pos', axis=1, inplace=True)

    # put a word type feature
    if word_choice == 'type':
        tags = get_tags()
        df['tags'] = df.apply(lambda row: tags[int(row.word)], axis=1)
        df['tags'] = df.tags.apply(categorise)
        df.tags = pd.Categorical(df.tags)
        df['tags'] = df.tags.cat.codes

    # remove missing values
    deal_with_missing(df)

    # standardise covariates
    covariates_to_std = ['surprisal', 'sent_pos', 'logwordfreq', 'wordlen', 'wordpos']
    df = standardize_features(df, covariates_to_std)

    # compute interaction terms
    if inter:
        df = create_interaction(df)

    # cast certain columns to int
    df.word, df.Participant = df.word.astype(int), df.Participant.astype(int)

    # filter out words as in Frank et al.
    idx = filter_words(sentences, words)
    df = df[df.word.isin(idx)]

    # select a number of participants and words
    df = select_subjects(df, subjects)
    df = select_words(df, num_words)
    return df


def main():
    parser = argparse.ArgumentParser(description='Processing data from Frank et al (2015). Please choose settings')
    parser.add_argument('--erp', default='N400', choices=['N400', 'all'],
                        help='ERP components to include: either \'N400\' component '
                             'or \'all\' components (default: N400)')
    parser.add_argument('--lm', type=str, default='ngram', choices=['ngram', 'lstm', 'gpt2'],
                        help='Language mode  to consider, either \'ngram\' or \'lstm\' or \'gpt\' (default: ngram)')
    parser.add_argument('--sj', type=int, default='24', help='Number of subjects to include (default: 24)')
    parser.add_argument('--nw', type=int, default='1931', help='Number of words to include (default: 1931)')
    parser.add_argument('--wd', type=str, default='type', choices=['words', 'type'],
                        help='Include all words (\'words\')'
                             ' or only word type (\'type\') '
                             '(default \'words\')')
    parser.add_argument('--sv', type=bool, default=True, help='Save dataframe object? (default: True)')
    parser.add_argument('--it', type=bool, default=False, help='Include interaction terms? (default: False)')

    args = parser.parse_args()

    erp_file = loadmat('data/stimuli_erp.mat')

    subject_range = np.arange(args.sj)

    df = generate_dataframe(erp_file, args.erp, args.lm, subject_range, args.nw, args.wd, args.it)

    if args.sv:
        df.to_csv(f'df_{args.lm}.csv', index=False)


if __name__ == '__main__':
    main()
