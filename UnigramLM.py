from collections import Counter

class UnigramLM:
    def __init__(self, voc):
        # self.voc = voc
        self.lm = Counter()
        self.valid = False
        self.sorted_start = []

    def add_token(self, token):
        self.valid = False
        if token in self.lm:
            self.lm[token] += 1
        else:
            self.lm[token] = 1

    def get_dist(self):
        if not self.valid:
            s = sum(self.lm.values())
            sk = sorted(self.lm.keys())
            self.sorted_start = sk
            self.p = [self.lm[k] / s for k in sk]
            self.valid = True

        return self.sorted_start, self.p