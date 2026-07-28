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
