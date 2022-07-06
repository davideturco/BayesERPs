from scipy.io import loadmat
import nltk
from nltk.tag import pos_tag_sents


# Import the universal tagset if not done already
# nltk.download('universal_tagset')


def load_sentences(file):
    """
    Loads the tokenised stimulus sentences into a list of lists of tokens.
    :param file: .mat file loaded via SciPy
    :return: list of lists
    """""
    all_sents = []
    for s in file['sentences'].flatten():
        sent = []
        for w in s[0].flatten():
            sent.append(w.item().replace('.', ''))
        all_sents.append(sent)

    return all_sents


def get_tags():
    erp_file = loadmat('data/stimuli_erp.mat')
    sentences = load_sentences(erp_file)
    tagged = pos_tag_sents(sentences, tagset='universal')
    tagged = [w for s in tagged for w in s]
    tags = [w[1] for w in tagged]
    # print(tags)
    return tags


if __name__ == '__main__':
    get_tags()
