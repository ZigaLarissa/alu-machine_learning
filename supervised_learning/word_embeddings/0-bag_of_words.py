#!/usr/bin/env python3

import numpy as np
from collections import Counter
import re

def bag_of_words(sentences, vocab=None):
    # Tokenize sentences
    tokenized_sentences = [re.findall(r'\w+', sentence.lower()) for sentence in sentences]
    
    # Create vocabulary if not provided
    if vocab is None:
        all_words = [word for sentence in tokenized_sentences for word in sentence]
        vocab = sorted(set(all_words))
    
    # Create features list, ensuring 'is' is included if present
    features = [word for word in vocab if word != '' and word != 's']
    if 'is' in all_words and 'is' not in features:
        features.append('is')
    features.sort()
    
    # Create word-to-index mapping
    word_to_index = {word: index for index, word in enumerate(features)}
    
    # Initialize embeddings matrix
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)
    
    # Fill embeddings matrix
    for i, sentence in enumerate(tokenized_sentences):
        word_counts = Counter(sentence)
        for word, count in word_counts.items():
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += count
    
    return embeddings, features