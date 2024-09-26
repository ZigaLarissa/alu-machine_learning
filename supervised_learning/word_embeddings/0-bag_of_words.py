#!/usr/bin/env python3
"""
Bag of words
"""
import numpy as np
from collections import Counter
import re


def bag_of_words(sentences, vocab=None):
    # Tokenize sentences and remove 's' at the end of words
    tokenized_sentences = [
        [word.rstrip('s') for word in re.findall(r'\w+', sentence.lower())]
        for sentence in sentences
    ]
    
    # Create vocabulary if not provided
    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))
    
    # Create features list
    features = vocab
    
    # Create word-to-index mapping
    word_to_index = {word: index for index, word in enumerate(features)}
    
    # Initialize embeddings matrix
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)
    
    # Fill embeddings matrix
    for i, sentence in enumerate(tokenized_sentences):
        word_counts = Counter(sentence)
        for word, count in word_counts.items():
            if word in word_to_index:
                embeddings[i, word_to_index[word]] = count
    
    return embeddings, features
