"""Evaulate a language model using a test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import math

from nltk.corpus import gutenberg


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    pattern = r'''(?ix)    # set flag to allow verbose regexps
          (?:sr\.|sra\.)
        | (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
        | \w+(?:-\w+)*        # words with optional internal hyphens
        | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
        | \.\.\.            # ellipsis
        | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
    '''

    tokenizer = RegexpTokenizer(pattern)
    root = '.'
    filename = 'eval.txt'
    corpus = PlaintextCorpusReader(root, filename, word_tokenizer=tokenizer)
    sents = list(corpus.sents())
    

    log_prob = model.log_prob(sents)
    e = model.cross_entropy(sents)
    p = model.perplexity(sents)

    print('Log probability: {}'.format(log_prob))
    print('Cross entropy: {}'.format(e))
    print('Perplexity: {}'.format(p))
