#!/usr/bin/env python3
"""
TF-IDF
"""

import numpy as np
from collections import Counter
import re
from math import log


def tf_idf(sentences, vocab=None):
    """
    tf-idf function
    """
    # Tokenize sentences
    tokenized_sentences = [
        re.findall(r'\w+', sentence.lower()) for sentence in sentences
        ]

    # Create vocabulary if not provided
    if vocab is None:
        all_words = [
            word for sentence in tokenized_sentences for word in sentence
            ]
        vocab = sorted(set(all_words))

    embeddings = []
    for sentence in tokenized_sentences:
        sentence_embedding = []
        for word in vocab:
            tf = sentence.count(word) / len(sentence) if sentence else 0
            df = sum([1 for s in tokenized_sentences if word in s])
            idf = log(len(sentences) / (df + 1))
            sentence_embedding.append(tf * idf)
        embeddings.append(sentence_embedding)

    return np.array(embeddings), vocab
