import numpy as np


def compute_mrr(gt_ids, ids, cutoff=1000):
    for i, id in enumerate(ids[:cutoff]):
        if id in gt_ids:
            return 1 / (i + 1)
    return 0


# compute recall from gts ids0 and ids1
def compute_recall(gt_ids, ids, cutoff=1000):
    return len(set(gt_ids).intersection(set(ids[:cutoff]))) / len(gt_ids)


def compute_hit(gt_ids, ids, cutoff=1000):
    # check that ids contains ALL gt_ids, and return 1 if so
    if set(gt_ids).issubset(set(ids[:cutoff])):
        return 1
    else:
        return 0


# compute precision
def compute_precision(gt_ids, ids, cutoff=1000):
    if len(ids[:cutoff]) == 0:
        return 0
    else:
        return len(set(gt_ids).intersection(set(ids[:cutoff]))) / len(ids[:cutoff])


def compute_rank_precision(gt_ids, ids, cutoff=1000):
    print(f"gt_ids: {gt_ids}, type: {type(gt_ids)}")
    print(f"ids: {ids}, type: {type(ids)}")
    if len(ids[:cutoff]) == 0:
        return 0
    else:
        divisor = min(len(gt_ids), cutoff)
        count = 0
        for i in range(min(cutoff, len(ids))):
            if ids[i] in gt_ids:
                count += 1
        # return len(set(gt_ids).intersection(set(ids[:cutoff]))) / divisor
        return count / divisor


def compute_f1score(gt_ids, ids, cutoff=1000):
    precision = compute_precision(gt_ids, ids, cutoff)
    recall = compute_recall(gt_ids, ids, cutoff)
    if precision + recall == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)


def get_exact_match(ret, gt):
    return set(ret) == set(gt)


def dcg(ratings):
    """Compute the Discounted Cumulative Gain (DCG) for a list of ratings."""
    return np.sum([(2**rating - 1) / np.log2(idx + 2) for idx, rating in enumerate(ratings)])


def ndcg(found_ratings, gt_ratings, k=None):
    """Compute the normalized Discounted Cumulative Gain (nDCG)."""
    if k:
        found_ratings = found_ratings[:k]
        gt_ratings = gt_ratings[:k]

    ideal_ratings = sorted(gt_ratings, reverse=True)

    dcg_score = dcg(found_ratings)
    idcg_score = dcg(ideal_ratings)

    return dcg_score / idcg_score if idcg_score > 0 else 0
