"""Milestone 5: SHAP explanations of the frozen protocol-v1 tree models (no training, no threshold changes).

Output spaces (checked by additivity on every call, see contributions()):
- LightGBM binary: raw margin = log-odds of the attack class; sigmoid(base + sum(SHAP)) == predict_proba[:, 1].
  Computed with LightGBM's native TreeSHAP (pred_contrib), which equals shap.TreeExplainer exactly (verified).
- Random Forest: probability of the attack class; base + sum(SHAP) == predict_proba[:, 1] (shap.TreeExplainer).
Both are path-dependent TreeSHAP (tree cover statistics from training, no background sample).
SHAP values describe how the model's output depends on its inputs; they are not causal effects.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

from ids.modeling import KEY, attack_score

TOL = 1e-6  # max |reconstructed - model output| accepted (probability and log-odds)

# Feature groups used to aggregate SHAP (sums of SHAP values within a group are themselves additive).
GROUPS = {
    "tcp_window": ["FWD Init Win Bytes", "Bwd Init Win Bytes"],
    "header_length": ["Fwd Header Length", "Bwd Header Length", "Fwd Seg Size Min"],
    "bwd_size": ["Total Length of Bwd Packet", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
                 "Bwd Packet Length Std", "Bwd Segment Size Avg", "Subflow Bwd Bytes", "Bwd Bytes/Bulk Avg"],
    "fwd_size": ["Total Length of Fwd Packet", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
                 "Fwd Packet Length Std", "Fwd Segment Size Avg", "Subflow Fwd Bytes", "Fwd Bytes/Bulk Avg"],
    "pkt_size_both": ["Packet Length Min", "Packet Length Max", "Packet Length Mean", "Packet Length Std",
                      "Packet Length Variance", "Average Packet Size"],
    "counts_direction": ["Total Fwd Packet", "Total Bwd packets", "Subflow Fwd Packets", "Subflow Bwd Packets",
                         "Fwd Act Data Pkts", "Down/Up Ratio", "Fwd Packet/Bulk Avg", "Bwd Packet/Bulk Avg"],
    "timing": ["Flow Duration", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total",
               "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
               "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Active Mean", "Active Std", "Active Max", "Active Min",
               "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Total TCP Flow Time"],
    "rates": ["Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s", "Fwd Bulk Rate Avg", "Bwd Bulk Rate Avg"],
    "flags": ["Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd RST Flags", "Bwd RST Flags",
              "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
              "URG Flag Count", "CWR Flag Count", "ECE Flag Count"],
    "protocol": ["Protocol", "ICMP Code", "ICMP Type"],
}
FEATURE_GROUP = {f: g for g, fs in GROUPS.items() for f in fs}


def model_inputs(model):
    """Names of the columns the estimator sees, in order (after train-fitted preprocessing)."""
    return list(model.named_steps["pre"].get_feature_names_out())


def contributions(model, X):
    """SHAP values of the attack class for raw-feature rows X.

    Returns (DataFrame rows x model inputs indexed like X, base value, output space name). Raises if the
    SHAP columns are not the estimator's inputs in order, or if additivity against the model's own
    attack score fails."""
    pre, est = model.named_steps["pre"], model.named_steps["est"]
    cols = model_inputs(model)
    Xt = pre.transform(X)
    if list(Xt.columns) != cols:
        raise ValueError("preprocessor output does not match its feature names")
    score, _ = attack_score(model, X)
    if isinstance(est, RandomForestClassifier):
        import shap
        ex = shap.TreeExplainer(est)
        k = list(est.classes_).index(1)
        sv = np.asarray(ex.shap_values(Xt, check_additivity=False))[:, :, k]
        base, space = float(np.ravel(ex.expected_value)[k]), "probability"
        err = np.abs(base + sv.sum(axis=1) - score).max(initial=0.0)
    else:
        booster = est.booster_
        if booster.feature_name() != [c.replace(" ", "_") for c in cols]:
            raise ValueError("booster feature order differs from preprocessor output")
        c = booster.predict(Xt, pred_contrib=True, num_iteration=est.best_iteration_)
        sv, bases = c[:, :-1], c[:, -1]
        if len(bases) and not np.all(bases == bases[0]):
            raise ValueError("LightGBM expected value is not constant")
        base, space = float(bases[0]) if len(bases) else float("nan"), "log_odds"
        raw = booster.predict(Xt, raw_score=True, num_iteration=est.best_iteration_)
        margin = base + sv.sum(axis=1)
        err = max(np.abs(margin - raw).max(initial=0.0), np.abs(1 / (1 + np.exp(-margin)) - score).max(initial=0.0))
    if err > TOL:
        raise ValueError(f"SHAP additivity failed: max error {err:.3g}")
    return pd.DataFrame(sv, columns=cols, index=X.index), base, space


def outcome(y, score, threshold):
    """TP/FP/TN/FN with the frozen rule: predicted attack when score >= threshold."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(score, dtype=float) >= threshold
    return np.select([y & p, ~y & p, ~y & ~p], ["TP", "FP", "TN"], "FN")


def stratified_sample(df, by, cap, seed, key=KEY):
    """Deterministic per-group sample: all rows of groups with <= cap rows, else a seeded subset of cap rows.

    Rows are sorted by `key` first and groups are visited in sorted order, so the result does not depend on the
    input row order. Returns the selected rows sorted by key."""
    d = df.sort_values(key, kind="mergesort").reset_index(drop=True)
    if d[by].isna().any().any():
        raise ValueError("group columns contain missing values")
    rng = np.random.default_rng(seed)
    idx = []
    for _, g in d.groupby(by, sort=True):
        idx.extend(g.index if len(g) <= cap else np.sort(rng.choice(g.index.to_numpy(), cap, replace=False)))
    return d.loc[sorted(idx)].reset_index(drop=True)


def attack_pairs(meta):
    """{(day, src, dst)} for successful-attack rows (y_binary == 1)."""
    a = meta[meta["y_binary"] == 1]
    return set(zip(a["day"], a["Src IP"], a["Dst IP"]))


def is_reverse_of_attack(meta, pairs):
    """True where (day, Dst IP, Src IP) is an attack (day, src, dst) pair, i.e. traffic in the reply direction."""
    return np.fromiter(((d, s, r) in pairs for d, s, r in zip(meta["day"], meta["Dst IP"], meta["Src IP"])),
                       bool, len(meta))


def importance(sv, mask=None):
    """Mean |SHAP| per feature (descending) over the rows in mask."""
    s = sv if mask is None else sv[np.asarray(mask)]
    return s.abs().mean().sort_values(ascending=False, kind="mergesort")


def group_shap(sv):
    """Per-row SHAP summed within feature groups (columns = groups present)."""
    return sv.T.groupby(sv.columns.map(lambda c: FEATURE_GROUP.get(c, "other"))).sum().T


def topk_overlap(a, b, k):
    """Share of the top-k features of ranking a that are also in the top-k of ranking b (index = features)."""
    return len(set(a.index[:k]) & set(b.index[:k])) / k


def rank_corr(a, b):
    """Spearman correlation of two importance vectors over their common features."""
    common = a.index.intersection(b.index)
    return float(spearmanr(a[common], b[common]).statistic)


def quantiles(values, groups, qs=(0.01, 0.1, 0.5, 0.9, 0.99, 0.999)):
    """n and quantiles of `values` per group (rows = groups)."""
    df = pd.DataFrame({"v": np.asarray(values, dtype=float), "g": np.asarray(groups)})
    q = df.groupby("g")["v"].quantile(list(qs)).unstack()
    q.columns = [f"q{c:g}" for c in q.columns]
    return pd.concat([df.groupby("g").size().rename("n"), q], axis=1)


def logit(p, eps=1e-15):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))
