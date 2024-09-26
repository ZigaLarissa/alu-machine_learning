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
    else:
        all_words = [
            word for sentence in tokenized_sentences for word in sentence
            ]

    # Create features list
    features = [word for word in vocab if word != '' and word != 's']

    # Create word-to-index mapping
    word_to_index = {word: index for index, word in enumerate(features)}

    # Calculate document frequency
    doc_frequency = Counter(
        word for sentence in tokenized_sentences for word in set(sentence)
        )

    # Calculate IDF
    idf = {word: log(
        len(sentences) / (doc_frequency[word] +1)
        ) for word in features}

    # Initialize embeddings matrix
    embeddings = np.zeros((len(sentences), len(features)))

    # Fill embeddings matrix
    for i, sentence in enumerate(tokenized_sentences):
        word_counts = Counter(sentence)
        for word, count in word_counts.items():
            if word in word_to_index:
                tf = count / len(sentence)
                tfidf = tf * idf[word]
                embeddings[i, word_to_index[word]] = tfidf

    return embeddings, features
