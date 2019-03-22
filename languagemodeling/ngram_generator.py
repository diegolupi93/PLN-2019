from collections import defaultdict
import random
import operator


class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self._n = model._n

        # compute the probabilities
        probs = defaultdict(dict)
        keys = model._count.keys()
        dic_filtered = [elem for elem in keys if len(elem)==self._n]
        for elem in dic_filtered:
            prev_tokens = elem[0:self._n - 1]
            token = elem[self._n - 1]
            prob = model.cond_prob(token,prev_tokens)
            lis = list(probs[prev_tokens].items())+[(token,prob)]
            probs[prev_tokens] = dict(lis + [(token,prob)])
        print(probs)
        self._probs = dict(probs)

        # sort in descending order for efficient sampling
        self._sorted_probs = sorted_probs = defaultdict(dict)
        for e, b in self._probs.items():
            sort = sorted(b.items(), key=operator.itemgetter(1))
            sorted_probs[e] = sort 
        self._sorted_probs = sorted_probs
    

    def generate_sent(self):
        """Randomly generate a sentence."""
        # WORK HERE!!

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # WORK HERE!!
