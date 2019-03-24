from collections import defaultdict
from numpy.random import uniform
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
        ## n-upla list of ngram.count keys
        dic_filtered = [elem for elem in keys if len(elem)==self._n]
        for elem in dic_filtered:
            prev_tokens = elem[0:self._n - 1]
            token = elem[self._n - 1]
            prob = model.cond_prob(token,prev_tokens)
            lis = list(probs[prev_tokens].items())+[(token,prob)]
            probs[prev_tokens] = dict(lis + [(token,prob)])
        self._probs = dict(probs)

        # sort in descending order for efficient sampling
        self._sorted_probs = sorted_probs = defaultdict(dict)
        for e, b in self._probs.items():
            sort = sorted(b.items(), key=operator.itemgetter(1))
            sorted_probs[e] = sort 
        self._sorted_probs = sorted_probs

    

    def generate_sent(self):
        """Randomly generate a sentence."""
        n = self._n
        prev_tokens = ('<s>',) * (n-1) if n > 1 else ()
        sent = []
        while True:
            prob = uniform()
            token = self.generate_token(prev_tokens)
            if token == '</s>':
                return sent
            sent.append(token)
            prev_tokens = (prev_tokens+(token,))[1:n+1]


    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        tokens = self._sorted_probs[prev_tokens]
        prob = uniform()
        for token, value in tokens:
            if prob < value:
                return token
            prob -= value
