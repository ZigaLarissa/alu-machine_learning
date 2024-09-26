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

    # Get vocabulary from the vectorizer if not provided
    vocab = vectorizer.get_feature_names_out()

    # Convert TF-IDF matrix to an array
    E = tfidf_matrix.toarray()

    return E, vocab
