#!/usr/bin/env python3
"""
N-gram BLEU Score Calculation
"""

from collections import Counter
import math


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.
    """

    # Step 1: Function to generate n-grams
    def get_ngrams(words, n):
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    # Step 2: Generate n-grams for the sentence and references
    cand_ngrams = get_ngrams(sentence, n)
    ref_ngrams = [get_ngrams(ref, n) for ref in references]

    # Step 3: Count n-grams in the candidate sentence
    cand_cnt = Counter(cand_ngrams)

    # Step 4: Get the max count for each n-gram from the references
    max_ref_cnt = Counter()
    for ref in ref_ngrams:
        ref_cnt = Counter(ref)
        for gram in ref_cnt:
            max_ref_cnt[gram] = max(max_ref_cnt[gram], ref_cnt[gram])

    # Step 5: Clip counts by reference max counts
    clipped_cnt = {
        gram: min(cand_cnt[gram], max_ref_cnt[gram]) for gram in cand_cnt
        }

    # Step 6: Calculate precision
    precision = sum(clipped_cnt.values()) / len(cand_ngrams)

    # Step 7: Calculate brevity penalty
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (abs(ref_len - len(sentence)), ref_len)
        )

    if len(sentence) > closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_ref_len / len(sentence))

    # Step 8: Calculate the final BLEU score
    bleu = bp * precision

    return bleu
