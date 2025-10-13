import numpy as np


def softmax(logits: np.ndarray, axis: int = -1):
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def compute_per_sample_metrics(logits: np.ndarray, labels: np.ndarray):
    """Compute key logit/softmax based metrics per sample.
    Args:
        logits: [N, C]
        labels: [N]
    Returns dict of arrays (length N each):
        correct, pred, top1, top2, margin, entropy, true_logit,
        true_prob, true_rank, conf_gap, logit_gap_true_pred
    """
    N, C = logits.shape
    pred = logits.argmax(1)
    correct = (pred == labels).astype(np.int32)
    # sort logits desc
    idx_sorted = np.argsort(-logits, axis=1)  # [N,C]
    top1_idx = idx_sorted[:, 0]
    top2_idx = idx_sorted[:, 1]
    top1 = logits[np.arange(N), top1_idx]
    top2 = logits[np.arange(N), top2_idx]
    margin = top1 - top2

    # true label stats
    true_logit = logits[np.arange(N), labels]
    # rank of true label (0-based)
    # position where idx_sorted == true_label
    true_rank = np.where(idx_sorted == labels[:, None])[1]

    # softmax / entropy
    probs = softmax(logits, axis=1)
    entropy = -(probs * np.log(probs + 1e-12)).sum(1)
    true_prob = probs[np.arange(N), labels]

    # confidence gap between top1 prob and second
    probs_sorted = np.take_along_axis(probs, idx_sorted, axis=1)
    prob_margin = probs_sorted[:, 0] - probs_sorted[:, 1]

    # gap between predicted logit and true logit (useful for wrong cases)
    logit_gap_true_pred = top1 - true_logit

    return {
        'correct': correct,
        'pred': pred,
        'top1': top1,
        'top2': top2,
        'margin': margin,
        'entropy': entropy,
        'true_logit': true_logit,
        'true_prob': true_prob,
        'true_rank': true_rank,
        'prob_margin': prob_margin,
        'logit_gap_true_pred': logit_gap_true_pred,
    }


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Compute ECE (Expected Calibration Error) for top-1 probs.
    probs: [N], max probability per sample
    labels: [N] ground truth, pred is argmax of probs
    """
    preds = probs.argmax(1) if probs.ndim == 2 else None
    if probs.ndim == 2:
        max_probs = probs.max(1)
        correct = (preds == labels)
    else:
        max_probs = probs
        correct = None  # not used
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m, M = bins[i], bins[i+1]
        mask = (max_probs >= m) & (max_probs < M)
        if mask.sum() == 0:
            continue
        acc_bin = correct[mask].mean() if correct is not None else 0
        conf_bin = max_probs[mask].mean()
        ece += (mask.sum() / len(max_probs)) * abs(acc_bin - conf_bin)
    return ece