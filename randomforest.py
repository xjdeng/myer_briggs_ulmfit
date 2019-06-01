from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from path import Path as path

def abspath(myfile):
    return path(__file__).dirname().abspath() + "/" + myfile

with open(abspath("english10000.txt"),'r') as f:
    words = f.read().split("\n")

wordset = set(words)
indexdict = {None: 0}
for i,w in enumerate(words):
    indexdict[w] = i
    
def get_distribution(txt):
    txtwords = txt.lower().split()
    dist = [0]*(len(words))
    for w in txtwords:
        if w in wordset:
            dist[indexdict[w]] += 1
    tot = sum(dist)
    if tot == 0:
        return [0]*(len(words))
    return [d/tot for d in dist]