# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:36:42 2017

@author: asus
"""

from gensim.models import word2vec
import logging

def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus("wiki_seg.txt")
    model = word2vec.Word2Vec(sentences, size=100, alpha=0.025, window=2, min_count=3, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1)

    # Save our model.
    model.save("D:\work\med250.model.bin")

    # To load a model.
    # model = word2vec.Word2Vec.load("your_model.bin")

if __name__ == "__main__":
    main()