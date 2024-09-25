#!/usr/bin/env python3

import numpy as np
import string
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Parameters:
    - sentences: List of sentences to analyze
    - vocab: List of the vocabulary words to use for the analysis.
     If None, all words within sentences will be used.

    Returns:
    - embeddings: numpy.ndarray of shape (s, f) containing the embeddings
    - features: List of the features (unique words) used for embeddings
    """

    # Tokenize the sentences by lowercasing and removing punctuation
    tokenized_sentences = []
    for sentence in sentences:
        # Remove punctuation using regex
        clean_sentence = re.sub(r'[^\w\s]', '', sentence)
        tokenized_sentences.append(clean_sentence.lower().split())

    # Create vocabulary if not provided
    if vocab is None:
        vocab = sorted(set(
            word for sentence in tokenized_sentences for word in sentence
            ))

    # Create a list of features (unique vocabulary words)
    features = vocab

    # Create the embeddings matrix
    s = len(sentences)
    f = len(features)

    # Initialize the embedding matrix with zeros
    embeddings = np.zeros((s, f), dtype=int)

    # Fill the embedding matrix
    for i, sentence in enumerate(tokenized_sentences):
        for word in sentence:
            if word in features:
                embeddings[i, features.index(word)] += 1

    return embeddings, features
