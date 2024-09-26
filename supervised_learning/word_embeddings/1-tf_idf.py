#!/usr/bin/env python3
"""
TF-IDF
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    tf-idf function
    """

    # Initialize TF-IDF vectorizer with optional vocabulary
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Fit and transform sentences
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # if vocab not, all words within sentence should be used
    vocab = [word for word in sentences]

    # Convert TF-IDF matrix to an array
    E = tfidf_matrix.toarray()

    return E, vocab
