"""Evaluation functions (no model training here).

Conventions:
- Positive class = attack (y_binary == 1). `score` = model probability/score for the attack class;
  higher means more likely attack. A row is predicted attack when score >= threshold.
- Undefined metrics are returned as None, never as 0 or 1:
  precision undefined when nothing is predicted positive; recall undefined when there are no positives;
  FPR undefined when there are no negatives; F1 = 2TP / (2TP + FP + FN), undefined only if that is 0/0;
  PR-AUC undefined without positives; ROC-AUC undefined unless both classes are present.
- Thresholds are chosen on validation data (select_threshold) and then applied unchanged to test data.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

UNSTABLE_SUPPORT = 30  # class-specific estimates with fewer test rows are flagged as unstable


def _div(a, b):
    return None if b == 0 else a / b


def confusion(y, pred):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return dict(tp=int((y & pred).sum()), fp=int((~y & pred).sum()), tn=int((~y & ~pred).sum()), fn=int((y & ~pred).sum()))


def binary_report(y, score, threshold, labels_original=None):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pred = score >= threshold
    c = confusion(y, pred)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    both = pos > 0 and neg > 0
    r = dict(threshold=float(threshold), n=len(y), support_attack=pos, support_benign=neg, **c,
             precision=_div(c["tp"], c["tp"] + c["fp"]), recall=_div(c["tp"], pos), fpr=_div(c["fp"], neg),
             f1=_div(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"]), false_positives=c["fp"],
             pr_auc=float(average_precision_score(y, score)) if pos > 0 else None,
             roc_auc=float(roc_auc_score(y, score)) if both else None,
             confusion_matrix=[[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    if labels_original is not None:
        r["per_original_label"] = per_label_rates(y, pred, labels_original)
    return r


def per_label_rates(y, pred, labels_original):
    """Detection rate for attack labels; alert (false-positive) rate for benign/Attempted labels."""
    labels_original = np.asarray(labels_original)
    out = {}
    for lab in np.unique(labels_original):
        m = labels_original == lab
        is_attack = bool(np.asarray(y)[m].max())
        k = int(np.asarray(pred)[m].sum())
        out[str(lab)] = dict(n=int(m.sum()), role="attack" if is_attack else "benign",
                             flagged=k, rate=k / m.sum(),
                             unstable=bool(m.sum() < UNSTABLE_SUPPORT))
    return out


def threshold_curve(y, score, thresholds):
    return [binary_report(y, score, t) | {"per_original_label": None} for t in thresholds]


def select_threshold(y_val, score_val, max_fpr):
    """Highest-recall threshold whose validation FPR <= max_fpr (ties -> higher threshold).

    Candidate thresholds are the distinct validation scores (plus +inf = flag nothing).
    Must only ever be called with validation data.
    """
    y = np.asarray(y_val).astype(bool)
    s = np.asarray(score_val, dtype=float)
    if y.all() or not y.any():
        raise ValueError("validation data needs both classes to select a threshold")
    cand = np.unique(s)[::-1]  # descending
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    # counts of rows with score >= cand[i]
    idx = np.searchsorted(-ss, -cand, side="right")
    tp = np.cumsum(yy)[idx - 1]
    fp = np.cumsum(~yy)[idx - 1]
    fpr = fp / (~y).sum()
    ok = fpr <= max_fpr
    if not ok.any():
        return float("inf")
    rec = tp / y.sum()
    best = np.flatnonzero(ok)[np.argmax(rec[ok])]  # argmax -> first = highest threshold among ties
    return float(cand[best])


def multiclass_report(y_true, y_pred, classes=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = sorted(set(y_true) | set(y_pred)) if classes is None else classes
    per = {}
    for k in classes:
        tp = int(((y_true == k) & (y_pred == k)).sum())
        fp = int(((y_true != k) & (y_pred == k)).sum())
        fn = int(((y_true == k) & (y_pred != k)).sum())
        sup = int((y_true == k).sum())
        per[str(k)] = dict(support=sup, precision=_div(tp, tp + fp), recall=_div(tp, sup),
                           f1=_div(2 * tp, 2 * tp + fp + fn), unstable=sup < UNSTABLE_SUPPORT)
    present = [v["f1"] for v in per.values() if v["support"] > 0]
    wts = [v["support"] for v in per.values() if v["support"] > 0]
    return dict(per_class=per,
                macro_f1=float(np.mean(present)) if present else None,  # over classes present in y_true
                weighted_f1=float(np.average(present, weights=wts)) if present else None,
                n=len(y_true))


def wilson(k, n, z=1.959964):
    """95 % Wilson score interval for k successes out of n; (None, None) when n == 0."""
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))
