from collections import Counter
# from .tokenization import get_document_words
from scipy.sparse import csr_matrix
import numpy as np
import nltk 

get_document_words = nltk.tokenize.word_tokenize

class Vocabulary:
    def __init__(self):
        """
        creates vocabulary of words from input string
        words are determined by get_document_words()
        """

        # self.tokens = list(set(get_document_words(text)))
        # self.index = dict(zip(self.tokens,range(len(self.tokens))))
        # self.inv_index = dict(zip(range(len(self.tokens)),self.tokens))
        # self.size = len(self.tokens)

        self.tokens = []
        self.index = dict()
        self.inv_index = dict()
        self.size = 0

    def find(self, tokens):
        ids = [self.index.get(token, -1) for token in tokens]
        return list(filter(lambda x: x != -1, ids))
        # return [self.index.get(token,-1) for token in tokens]

    def vectorize_query(self, query):
        words = get_document_words(query)
        wc = Counter(words)
        tc = sum(wc.values())
        col = np.array(self.find(wc.keys()))
        row = np.zeros(col.shape)
        val = np.array(list(wc.values()))
        # wc = csr_matrix((val, (row, col)), shape=(1, self.size))
        prob = csr_matrix((val / tc, (row, col)), shape=(1, self.size), dtype=np.float32)
        return prob

    def expand(self, doc, from_tokens=False):
        if not from_tokens:
            tokens = get_document_words(doc)
        else:
            tokens = doc
        new_tokens = list(set(tokens).difference(self.tokens))

        # create index expansion
        new_index = dict(zip(new_tokens, range(self.size, self.size + len(new_tokens))))
        new_inv_index = dict(zip(range(self.size, self.size + len(new_tokens)), new_tokens))

        # update vocabulary
        self.tokens += new_tokens
        self.index.update(new_index)
        self.inv_index.update(new_inv_index)
        old_size = self.size
        self.size = len(self.tokens)
        return old_size, self.size

    def get_word(self, w_ids):
        return [self.inv_index[w_id] for w_id in w_ids]
    
    def get_word_id(self, words):
        # print(words)
        return [self.index[w] for w in words]

    def sorted_tokens(self):
        return [self.inv_index[k] for k in sorted(self.inv_index.keys())]