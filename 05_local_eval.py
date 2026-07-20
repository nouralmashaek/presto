"""
Step 5 - Use this to sanity-check changes (normalization tweaks, fusion
weights, training epochs) WITHOUT spending a real leaderboard submission.

Implements the exact nDCG@10 formula from the competition brief:
  DCG@10  = sum( (2^grade_i - 1) / log2(i+1) )   for i = 1..10
  IDCG@10 = same sum for the ideal (best-first) ordering
  nDCG@10 = DCG@10 / IDCG@10

CAVEAT: the public training data only gives you binary-ish signal (one
ground-truth positive product per query, plus mined hard negatives) - not
the full graded 0-3 relevance scale the official test set uses. To build a
local validation set, hold out a slice of train_positives.parquet /
train_pairs_with_negatives.parquet yourself and treat the true positive as
grade 3, hard negatives as grade 0 (per the "-1 treated as 0" scoring
rule), and everything else as grade 0. This is an approximation of your
real leaderboard score, not a guarantee - use it for A/B comparisons
between your own approaches, not as an absolute number.
"""
import numpy as np


def dcg_at_10(ranked_grades: list) -> float:
    # rank i is 1-indexed in the formula; enumerate() gives 0-indexed i,
    # so the log uses (i + 2) to land on log2(rank + 1).
    return sum(
        (2 ** g - 1) / np.log2(i + 2)
        for i, g in enumerate(ranked_grades[:10])
    )


def ndcg_at_10(ranked_grades: list) -> float:
    ideal = sorted(ranked_grades, reverse=True)
    idcg = dcg_at_10(ideal)
    if idcg == 0:
        return 0.0
    return dcg_at_10(ranked_grades) / idcg


def evaluate(predictions: dict, ground_truth: dict) -> float:
    """
    predictions:   {query_id: [product_id, ...]}          your ranked top-10
    ground_truth:  {query_id: {product_id: grade}}         known relevance grades
    """
    scores = []
    for qid, pred_ids in predictions.items():
        grades = ground_truth.get(qid, {})
        # -1 (hard negative) and unlisted products both score as 0 per the rules
        ranked_grades = [max(grades.get(pid, 0), 0) for pid in pred_ids]
        scores.append(ndcg_at_10(ranked_grades))
    return float(np.mean(scores)) if scores else 0.0


if __name__ == "__main__":
    # Minimal smoke test
    preds = {"Q_1": ["P_a", "P_b", "P_c"]}
    truth = {"Q_1": {"P_a": 3, "P_b": 0, "P_c": 1}}
    print("Example nDCG@10:", evaluate(preds, truth))
