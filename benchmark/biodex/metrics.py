import numpy as np



def compute_recall(gt_ids, ids, cutoff=1000):
    return len(set(gt_ids).intersection(set(ids[:cutoff]))) / len(gt_ids)


def compute_precision(gt_ids, ids, cutoff=1000):
    if len(ids[:cutoff]) == 0:
        return 0
    else:
        return len(set(gt_ids).intersection(set(ids[:cutoff]))) / len(ids[:cutoff])


def compute_rank_precision(gt_ids, ids, cutoff=1000):
    if len(ids[:cutoff]) == 0:
        return 0
    else:
        divisor = min(len(gt_ids), cutoff)
        count = 0
        for i in range(min(cutoff, len(ids))):
            if ids[i] in gt_ids:
                count += 1
        return count / divisor
