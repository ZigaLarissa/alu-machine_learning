#!/usr/bin/env python3

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    word_sentences = [word for word in word_sentence if word.isalpha()]

    # Tokenize and clean the sentences
    tokenized_sentences = [re.findall(r'\b\w+\b', sentence.lower()) for sentence in word_sentences]
    
    
    # Build vocabulary if not provided
    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))
    
    # Create a list to store embeddings
    embeddings = []
    
    # Build the embedding matrix
    for sentence in tokenized_sentences:
        # Create a list for the current sentence embedding
        sentence_embedding = [sentence.count(word) for word in vocab]
        embeddings.append(sentence_embedding)
    
    # Convert embeddings to numpy array
    embeddings = np.array(embeddings)
    
    return embeddings, vocab
