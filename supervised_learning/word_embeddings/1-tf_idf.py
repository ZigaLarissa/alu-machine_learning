#!/usr/bin/env python3
"""
TF-IDF
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer


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

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Fit and transform sentences
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Convert TF-IDF matrix to arrays
    E = tfidf_matrix.toarray()

    return E, vocab
