"""
Step 2 (replaces manual review of synonym_candidates.json) - Run after
01_build_synonyms.py.

WHY THIS EXISTS: synonym_candidates.json groups full query phrasings by
the single product they point to. That's the wrong unit for a synonym
dictionary - generic words like "سماعات" (headphones) show up as a
"variant" under many different, unrelated products, and treating full
sentences as substitutable synonyms would let one product's name
overwrite unrelated queries. Hand-reviewing 72k phrase groups also isn't
feasible.

What DOES generalize safely across products is brand/dialect SPELLING of
the same word - e.g. ريلمي / ريالمي / ريلم / ريال مي / ريكلمي all just
mean "Realme," regardless of which product the query is about. This
script mines exactly that: it tokenizes every query + product name in
the training data, counts word frequency, and clusters words that are
near-identical by edit distance (blocked by first letter + similar
length, so it stays fast even over a large vocabulary).

Output: word_synonym_clusters.json - a much smaller, reviewable list of
{canonical_word: [variant1, variant2, ...]} clusters, sorted by cluster
size. Skim this (not synonym_candidates.json) to build synonyms.json.
"""
import json
import re
from collections import Counter

import pandas as pd
from rapidfuzz.distance import Levenshtein

from normalize import normalize_arabic

MIN_WORD_FREQ = 5       # ignore rare words/typos-once, too noisy to cluster reliably
MIN_WORD_LEN = 5         # short words (3-4 letters) are too ambiguous in Arabic -
                         # edit distance 1 covers too much ground and merges unrelated words

# Only cluster pure Arabic-letter tokens. SKU/model codes (B12, 100W, C01...)
# and Latin brand strings look like short alphanumeric strings that edit-distance
# clustering will happily merge together (100 <-> 1001 <-> 100W ...), which is
# pure noise, not a dialect spelling variant. Excluding them removed the vast
# majority of the garbage clusters seen in testing.
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
        # distance 1 for short/medium words; only allow 2 for longer words,
        # where a 2-char difference is proportionally still a small edit
        max_dist = 1
        # a real spelling variant/typo should be much rarer than the
        # correct/canonical form it's a mistake of. Two genuinely
        # different real words (e.g. الجيد "good" vs الحار "hot") tend to
        # both be reasonably frequent, so capping how common a "variant"
        # is allowed to be filters most of those false merges out.
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
    # only keep clusters that actually merged something, and require the
    # canonical form to be reasonably frequent (not a one-off typo that
    # happened to sort high by luck)
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
    # sort biggest clusters first - those are the highest-value merges to review
    clusters = dict(sorted(clusters.items(), key=lambda kv: -len(kv[1])))

    with open("word_synonym_clusters.json", "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

    total_variants = sum(len(v) for v in clusters.values())
    print(f"Found {len(clusters)} clusters covering {total_variants} spelling variants.")
    print("Wrote word_synonym_clusters.json - review this (NOT synonym_candidates.json)")
    print("and merge the good clusters + synonyms_seed.json into synonyms.json.")


if __name__ == "__main__":
    main()
