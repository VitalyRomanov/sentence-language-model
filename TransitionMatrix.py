import numpy as np
import scipy
import nltk

from Vocabulary import Vocabulary
from UnigramLM import UnigramLM

SENT_END = "<S_END>"




class TransitionMatrix:
    def __init__(self, corpus = ""):
        self.voc = Vocabulary()
        self.tm = scipy.sparse.dok_matrix((1000,1000), dtype=np.float32)

        self.add_from_text(corpus)
        self.start = UnigramLM(self.voc)
        self.valid = False
        self.sorted_tokens = []

    def add_from_text(self, text):
        self.valid = False
        tss = tokenize_corpus(text)

        for ts in tss:
            if len(ts) > 0:
                self.start.add_token(ts[0])

            self.voc.expand(ts, from_tokens=True)
            wids = self.voc.get_word_id(ts)
            
            maxwid = max(wids)

            if maxwid >= self.tm.shape[0]:
                self.tm.resize((maxwid + 1, maxwid + 1))

            grams = getNGrams(wids)

            for g in grams:
                self.tm[g] += 1

    def validate(self):
        if not self.valid:
            self.p = self.tm.tocsr()
            s = self.p.sum(axis=1)
            self.p /= s
            self.valid = True
            self.sorted_tokens = self.voc.sorted_tokens()


    def sample_start(self):
        t, p = self.start.get_dist()
        # print(t, p)
        return np.random.choice(t, p = p)

    def sample(self, t):
        self.validate()
        wid = self.voc.get_word_id([t])[0]
        p = np.squeeze(np.asarray(self.p[wid, :]).reshape(-1,1))
        return np.random.choice(self.sorted_tokens, p = p)


    def __str__(self):
        return repr(self.tm)



def tokenize_sentences(sentences):
    tokenized_sentences = [nltk.tokenize.word_tokenize(sentence) for sentence in sentences]
    for ts in tokenized_sentences:
        ts.append(SENT_END)

    return tokenized_sentences

def tokenize_corpus(corpus):
    tss = []
    for line in corpus.split("\n"):
        ts = tokenize_sentences(nltk.tokenize.sent_tokenize(line))

    tss.extend(ts)
    return tss

def getNGrams(tokenized_sentence):
    gram = []
    for i in range(len(tokenized_sentence) - 1):
        gram.append(tuple(tokenized_sentence[i:i+2]))

    return gram