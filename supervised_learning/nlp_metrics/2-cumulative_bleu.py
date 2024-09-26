#!/usr/bin/env python3
"""
Cumulative N-gram BLEU Score Calculation
"""

from collections import Counter
import math


def get_ngrams(words, n):
    """Helper function to generate n-grams."""
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]


def precision(references, sentence, n):
    """Calculates the precision for n-grams."""
    # Generate n-grams for the candidate sentence
    cand_ngrams = get_ngrams(sentence, n)

    # Generate n-grams for the references
    ref_ngrams = [get_ngrams(ref, n) for ref in references]

    # Count n-grams in the candidate sentence
    cand_cnt = Counter(cand_ngrams)

    # Get the max count for each n-gram from the references
    max_ref_cnt = Counter()
    for ref in ref_ngrams:
        ref_cnt = Counter(ref)
        for gram in ref_cnt:
            max_ref_cnt[gram] = max(max_ref_cnt[gram], ref_cnt[gram])

    # Clip counts by reference max counts
    clipped_cnt = {
        gram: min(cand_cnt[gram], max_ref_cnt[gram]) for gram in cand_cnt
        }

    # Calculate precision
    if len(cand_ngrams) == 0:
        return 0
    return sum(clipped_cnt.values()) / len(cand_ngrams)


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.
    """

    # Step 1: Calculate precision for each n-gram level (1 to n)
    precisions = []
    for i in range(1, n + 1):
        p = precision(references, sentence, i)
        precisions.append(p)

    # Step 2: Calculate the geometric mean of precisions
    if min(precisions) == 0:
        geometric_mean = 0
    else:
        geometric_mean = math.exp(sum(math.log(p) for p in precisions) / n)

    # Step 3: Calculate brevity penalty
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens,key=lambda ref_len: (
        abs(ref_len - len(sentence)), ref_len
        ))

    if len(sentence) > closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_ref_len / len(sentence))

    # Step 4: Calculate final cumulative BLEU score
    bleu_score = bp * geometric_mean

    return bleu_score
