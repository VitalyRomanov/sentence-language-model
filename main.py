import nltk
from sys import argv, exit
import os

from TransitionMatrix import TransitionMatrix, SENT_END


if len(argv) != 2:
    print("Usage: main.py TEXT_FILE_LOCATION" )
    exit()

TEXT_FILE = argv[1]

if not os.path.isfile(TEXT_FILE):
    print("{} is not a file".format(TEXT_FILE))
    exit



corpus = open(TEXT_FILE).read()

# corpus = "It was in July, 1805, and the speaker was the well-known Anna Pávlovna Schérer, maid of honor and favorite of the Empress Márya Fëdorovna. With these words she greeted Prince Vasíli Kurágin, a man of high rank and importance, who was the first to arrive at her reception. Anna Pávlovna had had a cough for some days. She was, as she said, suffering from la grippe; grippe being then a new word in St. Petersburg, used only by the elite. "

tm = TransitionMatrix()

# c = 0
for line in corpus.split("\n"):
    tm.add_from_text(line)
    # c += 1
    # if c > 1000: break


for _ in range(100):
    a = tm.sample_start() + " "
    prev = a; curr = ""
    for i in range(100):
        curr = tm.sample(prev)
        a += curr + " "
        prev = curr
        if curr == SENT_END:
            break
    print(a,"\n\n")
    


        

