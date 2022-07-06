from scipy.io import loadmat
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
import torch as t
from torch.nn.functional import log_softmax
import numpy as np
import matplotlib.pyplot as plt
import argparse
import lstm.data as data
import data_processing
from utils import correlation
from tag_sentences import load_sentences


def plot_corr_surprisal(surp, refer, word_arr, ax=None):
    """
    Plot scatter plot of transformer surprisal vs the ngram baseline by Frank et al (2015)
    :param surp: surprisal values from LSTM or GPT-2 (NumPy array)
    :param refer: reference surprisal values by ngrams (NumPy array)
    :param ax: Matplotlib axis object
    :return: Matplotlib axis object
    """
    if ax is None:
        ax = plt.gca()
    corr = correlation(refer, surp)
    for i, txt in enumerate(word_arr):
        ax.annotate(txt, (surp[i], refer[i]))
    ax.set_xlabel(f'current model ({args.model})')
    ax.set_ylabel('ngram (by Frank et al)')
    ax.set_title(f'All words, corr {corr}')
    ax.scatter(surp, refer)
    return ax


def estimate_surp_lstm(sentences):
    """Estimate surprisal from a LSTM model
        :param sentences: sentences (list of lists of strings)
        :return: surprisal values, reference ngram surprisal values (Numpy arrays)
        """
    corpus = data.Corpus('lstm/data/wikitext-2')

    model = t.load('lstm/model.pt', map_location=t.device('cpu'))
    model.eval()

    surprisal = []
    for sent in sentences:
        # add a point to calculate surprisal of the first word
        sent = ['.'] + sent
        hidden = model.init_hidden(1)
        tokens = t.LongTensor([corpus.dictionary.word2idx[w] if w in corpus.dictionary.word2idx
                               else corpus.dictionary.word2idx["<unk>"]
                               for w in sent])
        output, hidden = model(tokens.view(-1, 1), hidden)
        probs = log_softmax(output, dim=-1)

        for j, word in enumerate(sent[1:]):
            try:
                id = corpus.dictionary.word2idx[word]
            except KeyError:
                id = corpus.dictionary.word2idx["<unk>"]
            surp = - probs[j, id].item()
            surprisal.append(surp)

    # load ngram suprisal from Frank et al (2015)
    ngram = np.load('surp.npy')
    surp = np.asarray(surprisal)
    np.save('surp_lstm.npy', surp)
    return surp, ngram


def estimate_surp_gpt2(sentences, incl):
    """
    Estimate surprisal from a GPT-2 model
    :param sentences: sentences (list of lists of strings)
    :return: surprisal values, reference ngram surprisal values (Numpy arrays)
    """
    # load pretrained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large", add_prefix_space=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained("gpt2-large")  # , is_decoder=True)
    model.eval()
    surprisal = []
    cnt = 0
    to_signal = []

    # add unknown words to the vocabulary
    if incl:
        for sent in sentences:
            # add to the vocabulary tokens that were not
            for k, w in enumerate(sent):
                id = tokenizer.convert_tokens_to_ids(w)
                if tokenizer.convert_ids_to_tokens(id) == tokenizer.eos_token:
                    tokenizer.add_tokens([w], special_tokens=True)

        # resize model to the size of the new vocabulary (with words added)
        model.resize_token_embeddings(len(tokenizer))

    print('Word not in vocabulary:')
    for sent in sentences:
        # add a point to calculate surprisal of the first word
        sent = ['.'] + sent
        tokens = tokenizer(sent, return_tensors="pt", padding=True, is_split_into_words=True)
        inputs = tokens['input_ids']

        next_token_logits = model(inputs).logits

        # extract surprisal. Notice that we don't include the first word (which is a full stop)
        probs = log_softmax(next_token_logits, dim=-1)

        for j, word in enumerate(sent[1:]):
            id = tokenizer.convert_tokens_to_ids(word)
            surp = - probs[0, j, id].item()
            # surprisal.append(surp)

            if tokenizer.convert_ids_to_tokens(id) == tokenizer.eos_token:
                to_signal.append(cnt)
                surprisal.append(surp)
                print(word)
            else:
                surprisal.append(surp)
            cnt += 1

    # load ngram suprisal from Frank et al (2015)
    ngram = np.load('surp.npy')
    surp = np.asarray(surprisal)
    np.save('surp_gpt2.npy', surp)
    return surp, ngram


def main(args):
    # load sentences/words
    file = loadmat('data/stimuli_erp.mat')
    sentences = load_sentences(file)
    words = [w for sent in sentences for w in sent]

    idx = data_processing.filter_words(sentences, words)

    if args.model == 'lstm':
        surp, ngram = estimate_surp_lstm(sentences)
    elif args.model == 'gpt2':
        surp, ngram = estimate_surp_gpt2(sentences, args.voc)

    fig, ax = plt.subplots()
    ax = plot_corr_surprisal(surp[idx], ngram[idx], np.array(words)[idx], ax=ax)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate surprisal values using an LSTM or GPT-2.')
    parser.add_argument('--voc', type=bool, default=False, help='Include unknows words in GPT-2 vocabulary? (str.)')
    parser.add_argument('--model', choices=['lstm', 'gpt2'])
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    args = parser.parse_args()

    if t.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

    global device
    device = t.device("cuda" if args.cuda else "cpu")

    main(args)

# FOR GENERATING WORDS
# wordss = np.array(words)
# for i, j in zip(wordss[to_signal], to_signal):
#     ax.annotate(i, (surp[j], ngram[j]))
# ax.scatter(surp, ngram)
# ax.scatter(surp[to_signal], ngram[to_signal])
# plt.show()

# def main():
#     file = loadmat('data/stimuli_erp.mat')
#     sentences = load_sentences(file)
#     tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     model = AutoModelForCausalLM.from_pretrained("gpt2", is_decoder=True)
#     for i, sent in enumerate(sentences):
#         tokens = tokenizer(sent, return_tensors="pt", padding=True, is_split_into_words=True)
#         inputs = tokens['input_ids']
#         next_token_logits = model(inputs).logits[:,-1,:]
#         # filter
#         filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
#
#         # sample
#         probs = nn.functional.softmax(filtered_next_token_logits, dim=-1)
#         next_token = t.multinomial(probs, num_samples=1)
#
#         generated = t.cat([inputs, next_token], dim=-1)
#
#         resulting_string = tokenizer.decode(generated.tolist()[0])
#         print(resulting_string)
#         None
