# https://docs.python.org/3/library/collections.html
from collections import defaultdict
import math

def vocab(sents):
    voc = set()
    for sent in sents:
        voc = voc | set(sent)
    return voc


class LanguageModel(object):

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        return 0.0

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        return -math.inf

    def log_prob(self, sents):
        """Log-probability of a list of sentences.

        sents -- the sentences.
        """
        prob = 0
        for sent in sents:
            prob += sent_log_prob(sent)
        return prob

    def cross_entropy(self, sents):
        """Cross-entropy of a list of sentences.

        sents -- the sentences.
        """
        # cross-entropy = log-probability / "cantidad de palabras"  
        log_pr = log_prob(sents)
        voc = vocab(sents)
        return log_pr/float(len(voc))

    def perplexity(self, sents):
        """Perplexity of a list of sentences.

        sents -- the sentences.
        """
        #perplexity = 2 ** (- cross-entropy)
        return 2**cross_entropy(sents)


class NGram(LanguageModel):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self._n = n

        count = defaultdict(int)

        for sent in sents:
            sent.append('</s>')
            sent = ['<s>'] * (n-1) + sent
            for i in range(len(sent)-n+1):
                ngram = tuple(sent[i:i+n])
                ngram2 = tuple(sent[i:i+n-1])
                count[ngram] += 1
                count[ngram2] += 1

        self._count = dict(count)

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        tupl = prev_tokens + (token,) if prev_tokens else (token,)
        count = self._count.get(tupl, 0)
        count2 = self._count.get(prev_tokens,1) if prev_tokens else self._count[()]
        return count/float(count2)


    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        n = self._n
        sent.append('</s>')
        sent = ['<s>'] * (n-1) + sent
        prob = 1
        for i in range(len(sent)-n+1):
            token = sent[i+n-1]
            prev_tokens = tuple(sent[i:i+n-1])
            prob *= self.cond_prob(token,prev_tokens)  
        return prob
        

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        n = self._n
        sent.append('</s>')
        sent = ['<s>'] * (n-1) + sent
        prob = 0
        for i in range(len(sent)-n+1):
            token = sent[i+n-1]
            prev_tokens = tuple(sent[i:i+n-1])
            cond = self.cond_prob(token,prev_tokens)
            if cond == 0:
                return -math.inf
            prob += math.log(cond,2)
        return prob


class AddOneNGram(NGram):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        # call superclass to compute counts
        super().__init__(n, sents)

        # compute vocabulary
        self._voc = voc = set()
        voc = vocab(sents)
        self._V = len(voc)  # vocabulary size

    def V(self):
        """Size of the vocabulary.
        """
        return self._V

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        
        tupl = prev_tokens + (token,) if prev_tokens else (token,)
        count = self._count.get(tupl, 0)
        count2 = self._count.get(prev_tokens,0) if prev_tokens else self._count[()]
        return (count+1)/float(count2+self.V())



class InterpolatedNGram(NGram):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self._n = n

        if gamma is not None:
            # everything is training data
            train_sents = sents
        else:
            # 90% training, 10% held-out
            m = int(0.9 * len(sents))
            train_sents = sents[:m]
            held_out_sents = sents[m:]

        print('Computing counts...')
        # COMPUTE COUNTS FOR ALL K-GRAMS WITH K <= N
        count = defaultdict(int)
        for sent in sents:
            sent.append('</s>')
            sent = ['<s>'] * (n-1) + sent
            for i in range(len(sent)-n+1):
                for j in range(n+1):
                    ngram = tuple(sent[i:i+n-j])
                    count[ngram] += 1
                if i == len(sent)-n:
                    for l in range(1,n):
                        ngram = tuple(sent[i+l:i+n])
                        count[ngram] += 1
        self._count = dict(count)

        # compute vocabulary size for add-one in the last step
        self._addone = addone
        if addone:
            print('Computing vocabulary...')
            self._voc = voc = set()
            voc = vocab(sents)
            self._V = len(voc)

        # compute gamma if not given
        if gamma is not None:
            self._gamma = gamma
        else:
            print('Computing gamma...')
            # WORK HERE!!
            # use grid search to choose gamma

    def count(self, tokens):
        """Count for an k-gram for k <= n.

        tokens -- the k-gram tuple.
        """
        return self._count.get(tokens, 0)

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # WORK HERE!!
