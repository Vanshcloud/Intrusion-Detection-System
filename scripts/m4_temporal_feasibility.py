"""Feasibility of a label-independent chronological evaluation (Milestone 4, item 19). No training.

Candidate splits whose boundaries do not use labels:
  C1 per-day cut: within each day, the time at which 70 % of that day's flows have started.
  C2 week cut:    train Monday-Wednesday, test Thursday-Friday.
  C3 week cut:    train Monday-Thursday, test Friday.
Labels are used only to *report* what each side contains.

Pre-declared feasibility criterion (for a supervised benchmark of the same question as protocol v1):
every successful-attack family present in the test side must have >= 30 rows in the train side,
and the test side must contain both classes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids.data import GENERATED, PROCESSED  # noqa: E402

MIN_TRAIN = 30


def main():
    d = pd.read_parquet(PROCESSED / "dataset_v1.parquet", columns=["day", "t_start", "family", "y_binary"])
    cut = d.groupby("day")["t_start"].transform(lambda t: np.quantile(t, 0.7))
    sides = {"C1_per_day_70pct": np.where(d["t_start"] < cut, "train", "test"),
             "C2_mon-wed_vs_thu-fri": np.where(d["day"].isin(["monday", "tuesday", "wednesday"]), "train", "test"),
             "C3_mon-thu_vs_fri": np.where(d["day"] == "friday", "test", "train")}
    rows, verdict = [], []
    for name, side in sides.items():
        t = d.assign(side=side).groupby(["family", "side"]).size().unstack(fill_value=0)
        for fam, r in t.iterrows():
            rows.append({"candidate": name, "family": fam, "train": int(r.get("train", 0)), "test": int(r.get("test", 0))})
        att = t.drop(index="BENIGN")
        in_test = att[att.get("test", 0) > 0]
        bad = in_test[in_test.get("train", 0) < MIN_TRAIN].index.tolist()
        verdict.append({"candidate": name, "attack_families_in_test": len(in_test),
                        "families_in_test_without_enough_train": ", ".join(bad) or "none",
                        "test_has_both_classes": bool(d.loc[side == "test", "y_binary"].nunique() == 2),
                        "feasible_for_v1_question": not bad and d.loc[side == "test", "y_binary"].nunique() == 2})
    pd.DataFrame(rows).to_csv(GENERATED / "m4_temporal_feasibility_counts.csv", index=False)
    pd.DataFrame(verdict).to_csv(GENERATED / "m4_temporal_feasibility.csv", index=False)
    print(pd.DataFrame(verdict).to_string())


if __name__ == "__main__":
    main()
