import json
import re
from collections import Counter

import pandas as pd
from rapidfuzz.distance import Levenshtein

from normalize import normalize_arabic

MIN_WORD_FREQ = 5       
MIN_WORD_LEN = 5        
                        

_ARABIC_ONLY = re.compile(r"^[\u0600-\u06FF]+$")


def collect_word_freq() -> Counter:
    freq = Counter()
    for path, cols in [
        ("data/train_positives.parquet", ["user_query", "positive_product_name"]),
        ("data/train_pairs_with_negatives.parquet",
         ["user_query", "positive_product_name", "negative_product_name"]),
    ]:
        df = pd.read_parquet(path, columns=cols)
        for col in cols:
            for text in df[col].dropna():
                if not isinstance(text, str) or '"' in text:
                    continue  # skip leaked JSON-key fragments, see 01_build_synonyms.py
                for word in normalize_arabic(text).split():
                    if len(word) >= MIN_WORD_LEN and _ARABIC_ONLY.match(word):
                        freq[word] += 1
    return freq


def cluster_words(freq: Counter) -> dict:
    """Greedy clustering: process words most-frequent first, so the
    canonical form of each cluster is always the most common spelling.
    Blocks comparisons by the first 2 characters (a much tighter block
    than length alone - Arabic dialect/spelling variants almost always
    share their opening letters, and this cuts down false merges between
    same-length-but-unrelated words significantly)."""
    words_by_prefix = {}
    for w in freq:
        words_by_prefix.setdefault(w[:2], []).append(w)

    assigned = set()
    clusters = {}
    for word, _ in freq.most_common():
        if word in assigned:
            continue
        clusters[word] = []
        assigned.add(word)
       
        max_dist = 1
        
        variant_freq_cap = max(3, int(freq[word] * 0.2))
        for other in words_by_prefix.get(word[:2], []):
            if other in assigned or other == word:
                continue
            if freq[other] > variant_freq_cap:
                continue
            if abs(len(other) - len(word)) > max_dist:
                continue
            if Levenshtein.distance(word, other) <= max_dist:
                clusters[word].append(other)
                assigned.add(other)

    return {
        canon: sorted(variants)
        for canon, variants in clusters.items()
        if variants and freq[canon] >= MIN_WORD_FREQ
    }


def main():
    print("Counting word frequencies across training data...")
    freq = collect_word_freq()
    print(f"Vocabulary size: {len(freq)} words (freq >= 1)")

    print("Clustering near-identical spellings...")
    clusters = cluster_words(freq)
 
    clusters = dict(sorted(clusters.items(), key=lambda kv: -len(kv[1])))

    with open("word_synonym_clusters.json", "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

    total_variants = sum(len(v) for v in clusters.values())
    print(f"Found {len(clusters)} clusters covering {total_variants} spelling variants.")
    print("Wrote word_synonym_clusters.json - review this (NOT synonym_candidates.json)")
    print("and merge the good clusters + synonyms_seed.json into synonyms.json.")


if __name__ == "__main__":
    main()
