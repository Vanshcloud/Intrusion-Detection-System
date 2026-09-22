import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from ids.metrics import binary_report, multiclass_report, select_threshold


def test_binary_report_matches_sklearn_and_counts():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    s = np.array([.1, .4, .6, .7, .2, .9, .3, .8])
    r = binary_report(y, s, 0.5, labels_original=np.array(["B", "B", "B", "X", "X", "Y", "B", "Y"]))
    pred = s >= 0.5
    assert r["precision"] == precision_score(y, pred) and r["recall"] == recall_score(y, pred)
    assert r["f1"] == pytest.approx(f1_score(y, pred)) and r["pr_auc"] == average_precision_score(y, s)
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (3, 1, 3, 1) and r["fpr"] == 0.25
    assert r["per_original_label"]["X"]["rate"] == 0.5 and r["per_original_label"]["B"]["flagged"] == 1
    assert r["per_original_label"]["X"]["unstable"]


def test_undefined_metrics_are_none_not_numbers():
    r = binary_report(np.zeros(4, int), np.array([.1, .2, .3, .4]), 0.9)  # no attacks, nothing flagged
    assert r["recall"] is None and r["precision"] is None and r["pr_auc"] is None and r["roc_auc"] is None
    assert r["f1"] is None and r["fpr"] == 0.0
    r = binary_report(np.ones(3, int), np.array([.1, .2, .3]), 0.9)
    assert r["fpr"] is None and r["recall"] == 0.0 and r["f1"] == 0.0


def test_select_threshold_respects_fpr_budget():
    y = np.array([0] * 10 + [1] * 5)
    s = np.array([.05, .1, .15, .2, .25, .3, .35, .4, .45, .95] + [.5, .6, .7, .96, .97])
    t = select_threshold(y, s, max_fpr=0.0)
    assert t == 0.96  # 0.95 is a benign score; any lower threshold admits a false positive
    t = select_threshold(y, s, max_fpr=0.1)
    assert t == 0.5
    r = binary_report(y, s, t)
    assert r["fpr"] <= 0.1 and r["recall"] == 1.0
    assert select_threshold(y, np.where(y == 1, 0.0, 1.0), max_fpr=0.0) == float("inf")


def test_multiclass_report_undefined_and_macro():
    yt = np.array(["a", "a", "b", "b"])
    yp = np.array(["a", "a", "a", "c"])
    r = multiclass_report(yt, yp)
    assert r["per_class"]["b"]["precision"] is None and r["per_class"]["b"]["recall"] == 0.0
    assert r["per_class"]["c"]["recall"] is None and r["per_class"]["c"]["f1"] == 0.0
    assert r["macro_f1"] == pytest.approx(np.mean([0.8, 0.0]))  # only classes present in y_true
