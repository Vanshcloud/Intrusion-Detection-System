# Project Plan — Explainable ML-Based Network Intrusion Detection (CIC-IDS2017)

Status: **Milestone 5 complete (SHAP explanations and error analysis of the frozen Milestone 4 models; nothing retrained); approved 2026-09-23.** Results: `reports/MODEL_EVALUATION.md` (M4), `reports/EXPLAINABILITY_ANALYSIS.md` and `MODEL_CARD.md` (M5). No dashboard or multiclass work yet.
Milestone record: M1 plan (this document) → M2 audit (`reports/DATA_AUDIT.md`, `reports/LEAKAGE_AUDIT.md`) → M3 protocol (`reports/EXPERIMENT_PROTOCOL.md`) → M4 models (`configs/model_selection_v1.json` addendum A, `configs/m4_frozen.json`, `reports/MODEL_EVALUATION.md`) → M5 explainability (`reports/EXPLAINABILITY_ANALYSIS.md`, `MODEL_CARD.md`). Where later milestones changed an assumption below, a dated note says so; the authoritative current design is `reports/EXPERIMENT_PROTOCOL.md`.
Every numeric value in this document is either (a) quoted from a cited source, or (b) marked **TBD — determined after running the experiment**.

Verification date for all web sources: 2026-09-22.

---

## 1. Objective

Build a leakage-aware, explainable supervised intrusion-detection pipeline on one public benchmark (CIC-IDS2017), evaluated with security-relevant metrics and analysed per attack category. This is a research/portfolio project, not a production IDS.

## 2. Research questions

- **RQ1.** How reliably can supervised ML separate benign from malicious flows in CIC-IDS2017 when the evaluation is designed to reduce temporal, duplicate and identifier leakage?
- **RQ2.** How does performance vary per attack category, and which categories are hard?
- **RQ3.** How much of the headline performance under a random split is lost under a leakage-aware split? (Quantifies optimism; answers "is the usual 99% real?")
- **RQ4.** Do SHAP attributions point to plausible traffic behaviour, or to dataset artefacts (e.g. environment-specific values)?
- **RQ5 (optional, Milestone 12).** How does a detector perform on an attack category excluded from training?

## 3. Environment (inspected)

| Item | Observed |
|---|---|
| Repository | Empty directory, not a git repo |
| System Python | 3.14.0 (Homebrew); only `numpy` installed |
| Env tooling | `uv` available; no conda |
| Hardware | Apple Silicon (arm64), 10 cores, 16 GB RAM, ~156 GB free disk |

Plan: create a `uv`-managed virtualenv pinned to **Python 3.12** (all planned packages declare `requires_python` ≤ 3.12; 3.12 avoids relying on fresh 3.14 wheels). Exact versions pinned in `requirements.txt` at Milestone 2. *(Done: `.venv`, Python 3.12.)*

## 4. Dataset

### 4.1 Official source (verified)
- Page: <https://www.unb.ca/cic/datasets/ids-2017.html> (Canadian Institute for Cybersecurity, UNB).
- Download: `http://cicresearch.ca/CICDataset/CIC-IDS-2017/` → redirects to HTTPS and serves a **registration form** (first/last name, email, organisation, job title, country). It cannot be scripted without submitting personal data; the user must do it manually.
- Offered artefacts (quoted from page): full PCAPs, `GeneratedLabelledFlows.zip`, `MachineLearningCSV.zip`.
- Licence text on the page: data are "publicly available for researchers"; users "should cite" Sharafaldin et al. (ICISSP 2018). **No explicit redistribution grant is stated on the CIC-IDS2017 page** → we will not commit any data.

### 4.2 Collection structure (quoted from the official page)
Capture from 09:00 Mon 3 July 2017 to 17:00 Fri 7 July 2017; 25 simulated users (B-Profile) over HTTP, HTTPS, FTP, SSH, email; flows extracted with CICFlowMeter.

| Day | Attacks (official page times) |
|---|---|
| Mon 3 Jul | Benign only |
| Tue 4 Jul | FTP-Patator 9:20–10:20; SSH-Patator 14:00–15:00 |
| Wed 5 Jul | DoS slowloris 9:47–10:10; Slowhttptest 10:14–10:35; Hulk 10:43–11:00; GoldenEye 11:10–11:23; Heartbleed 15:12–15:32 |
| Thu 6 Jul | Web Brute Force 9:20–10:00; XSS 10:15–10:35; SQL Injection 10:40–10:42; Infiltration (afternoon) |
| Fri 7 Jul | Botnet ARES 10:02–11:02; Port Scan 13:55–15:27; DDoS LOIT 15:56–16:16 |

**Consequence for validation:** each attack type appears on exactly one day, inside a short, contiguous time window. A pure day-based train/test split therefore cannot evaluate standard supervised detection of most attack types (the test-day attacks would never be seen in training). This shapes §7.

### 4.3 Known data-quality problems (verified from primary sources)
1. **Engelen, Rimmer & Joosen (IEEE SPW 2021)** — abstract: "a series of problems with traffic generation, flow construction, feature extraction and labelling"; "more than 20 percent of original traffic traces are reconstructed or relabelled". Their project page documents CICFlowMeter fixes (TCP flow terminated after a single FIN; RST packets ignored) and introduces "X - Attempted" labels for payload-less flows of payload-reliant attacks.
2. **Liu, Engelen, Lynar, Essam & Joosen (IEEE CNS 2022)** — abstract: "a large number of previously undocumented errors … in attack orchestration, feature generation, documentation, and labeling"; publishes a fully re-created dataset with public labelling logic. Their per-attack documentation states, e.g., that DoS Hulk is mis-implemented (`Connection: close`) and that nearly the whole Hulk attack can be filtered with four conditions on `Total Length of Bwd Packets` — a direct warning about **shortcut features**.
3. **Lanvin et al. (CRiSIS 2022, LNCS 2023)** — abstract: flaws in labelling and traffic capture "such as packet misorder, packet duplication and attack that were performed but not correctly labelled".
4. **Dube (J. Computer Virology and Hacking Techniques, 2023)**, "Faulty use of the CIC-IDS 2017 dataset in information security research" — title/venue/DOI verified; abstract not accessible to us, so **its specific claims are not used** in this project until the paper is read.

### 4.4 Dataset version — DECISION REQUIRED

| Option | Pros | Cons |
|---|---|---|
| **A. Improved CIC-IDS2017 (Liu et al., CNS 2022)** — *recommended* | Re-extracted from the official PCAPs with fixed CICFlowMeter; labelling logic public; free direct download (no form); identifier columns present for leakage auditing; per-day files | Third-party derivative; its own licence/terms are **not stated** on the download page (unresolved); less common in older literature, so results are not directly comparable to papers using the original CSVs |
| B. Original `MachineLearningCSV.zip` (CIC) | Official; most-cited | Documented labelling and flow-construction errors (§4.3); requires registration form; schema **not verified by us** (behind form) |

Both are CIC-IDS2017 (same PCAPs, same scenario); Option A is not a switch to a different dataset. Recommendation: **A as the primary dataset**, cite both the original authors and Liu et al., and state explicitly in README/DATA_CARD which version was used.

### 4.5 What has been verified about Option A without downloading it
Using HTTP range requests on the zip's central directory (≈64 KB fetched, not the full file):

| File (in `CICIDS2017_improved.zip`, 343,549,013 bytes) | Uncompressed bytes |
|---|---:|
| monday.csv | 207,875,155 |
| tuesday.csv | 178,397,720 |
| wednesday.csv | 291,290,505 |
| thursday.csv | 189,519,159 |
| friday.csv | 285,188,226 |

- All five files have an identical header of **91 columns** (verified by reading only the first line of each).
- Columns include identifiers/metadata: `id`, `Flow ID`, `Src IP`, `Src Port`, `Dst IP`, `Dst Port`, `Protocol`, `Timestamp`; target-related: `Label`, `Attempted Category`; plus CICFlowMeter flow statistics (incl. added `ICMP Code`, `ICMP Type`, `Total TCP Flow Time`, `Fwd RST Flags`, `Bwd RST Flags`).
- Authors' instruction (download page): "Attempted" flows must **not** be treated as a separate ML class; "when in doubt" relabel them Benign (identified by `Attempted Category != -1`).
- Row counts, label values, class distribution, missing/infinite values, duplicates: **TBD — determined after running the audit (Milestone 2).** (The WTMC 2021 page publishes counts for its *2021* version; those are not assumed to hold for the 2022 version.)

## 5. Proposed methodology (pipeline)

```
download (manual/scripted) → SHA-256 verify → CSV → Parquet (typed, per day)
  → audit (DATA_AUDIT.md) → label policy (Attempted→Benign; ablation: drop)
  → leakage audit (LEAKAGE_AUDIT.md) → split (§7)
  → sklearn Pipeline fitted on train only → models (§9) → metrics (§10)
  → threshold analysis → per-attack analysis → SHAP → dashboard
```

## 6. Leakage risks identified (to be tested, not assumed)

| Risk | Why it is a risk | Planned test / handling |
|---|---|---|
| Identifier columns (`id`, `Flow ID`, `Src IP`, `Dst IP`, `Src Port`, `Timestamp`) | Labelling logic in Liu et al. is literally defined by IP + port + time window, e.g. attacker `172.16.0.1` → victim `192.168.10.50` for FTP/SSH-Patator and Hulk. A model given these memorises the scenario | Excluded from features; retained only for grouping/splitting/audit. Measure how well IP/time alone predict the label (quantifies the shortcut) |
| `Dst Port` | Legitimate service signal, but attacks target fixed ports (21, 22, 80 per labelling logic) | Ablation: with vs without |
| `Attempted Category` | Derived from the labelling process → direct target leakage | Never a feature; unit test asserts absence |
| Environment-specific values (e.g. `FWD Init Win Bytes`, `Bwd Init Win Bytes`) | May encode the attacker/victim OS/tool rather than behaviour (hypothesis, not established) | SHAP + ablation; report whether performance depends on them |
| Tool-signature shortcuts (e.g. Hulk vs `Total Length of Bwd Packet`, per Liu et al.) | Near-perfect separation from an artefact inflates results | Single-feature separability scan per attack; report |
| Exact / near-duplicate rows | Same feature vector in train and test inflates metrics; conflicting labels on identical vectors indicate label noise | Count duplicates within/between splits and conflicting-label duplicates; dedup policy documented |
| Temporal autocorrelation | Flows of one attack burst are near-identical; random split puts siblings in train and test | Chronological split with an embargo gap (§7) and random split only as comparison |
| Preprocessing leakage | Scaler/imputer/selection fitted on all data | Everything inside an sklearn `Pipeline` fitted on train; test asserts fit only sees train indices |
| Resampling leakage | SMOTE before split | Not planned by default; if used, only inside training folds |

## 7. Validation strategy (proposed; finalised after the audit)

> **Milestone 2 update:** the audit tested the design below against the data. The plain 70/30 split does not reduce duplicate overlap and the embargo discards most short-episode test rows. The revised proposal (training-side purge + novel-vector test subset) is in `reports/LEAKAGE_AUDIT.md` §10.
>
> **Milestone 3 update (approved and implemented):** chronological split within (day, label) strata, ~56/14/30 train/val/test, training- and validation-side purging, no embargo, novel-vector test subset per feature set, random split as comparison. It uses label-defined strata, so it is **not** a deployment simulation. See `reports/EXPERIMENT_PROTOCOL.md`. The text below is the original Milestone 1 hypothesis, kept for the record.

Information required before fixing the split (all **TBD** at Milestone 2): per-day row counts; per-attack counts and exact time spans in `Timestamp`; timestamp timezone and monotonicity; duplicate structure; how many flows per attack remain after the Attempted policy.

**Primary — per-episode chronological split with embargo.**
For each attack episode (and for benign traffic within each day), order by `Timestamp`; earliest ~70 % → train, latest ~30 % → test, with an embargo gap (e.g. drop flows within N seconds of the boundary; N chosen from observed flow durations, including the 120 s flow timeout noted by Liu et al.). Rationale: every attack type is seen in training, but test flows come from a *later* period, so the model cannot interpolate between temporally adjacent siblings. Hyper-parameters tuned on a chronological validation slice carved from the training portion only.
*Limitation:* train and test still share the same hosts, tools and session; this measures within-scenario temporal generalisation, not cross-network generalisation.

**Secondary — stratified random split** (same models). Reported only as a comparison to quantify optimism (RQ3), never as evidence of generalisation.

**Robustness (Milestone 12, if justified) — leave-one-attack-family-out** (possible because each family lives in its own time window). Described as "evaluation on an attack category excluded from training", not "zero-day detection".

**Small-class caveat.** Some classes are expected to be tiny (the 2021 version lists Heartbleed 11 and SQL Injection 12 effective flows; 2022 counts TBD). Per-class metrics for such classes will be reported with counts and flagged as unreliable; they may be merged into a family (e.g. "Web Attack") for multiclass evaluation — decided after the audit, documented either way.

## 8. Class imbalance

Measure first (TBD). Default: class weighting (`class_weight` / `scale_pos_weight`) plus threshold analysis. Resampling only if weighting is shown insufficient, only on training data.

## 9. Models (planned)

> **Milestone 4 update:** all three trained. Selection rule and search spaces in `configs/model_selection_v1.json` (addendum A to protocol v1, written before training); selected model LightGBM (tie-break over Random Forest on validation PR-AUC). Logistic regression uses a stateless signed log1p before scaling (documented configuration choice, not a protocol change). Key result: validation false-alarm rates did not transfer to the chronological test period (see MODEL_EVALUATION §8.1).

| # | Model | Why |
|---|---|---|
| 1 | Logistic Regression (standardised, class-weighted) | Transparent linear baseline; shows how much non-linearity buys |
| 2 | Random Forest | Standard strong tabular baseline; exact TreeSHAP |
| 3 | LightGBM (one gradient-boosting model only) | Fast on ~millions of rows on a laptop; exact TreeSHAP supported by `shap` (compatibility re-verified at Milestone 11) |

Tasks: binary (benign vs attack) as primary; multiclass (attack category) as secondary. Final model chosen from measured validation results, not in advance.

## 10. Metrics

- **Binary:** precision, recall (attack detection rate), F1, specificity, **FPR**, confusion matrix, PR-AUC (primary threshold-free metric given imbalance), ROC-AUC (secondary).
- **Multiclass:** per-class precision/recall/F1, macro-F1, weighted-F1, confusion matrix.
- **Per-attack (binary detector):** n, recall, false negatives, for **every** attack category in the test set.
- **Threshold analysis:** recall vs FPR curve; operating point chosen by a stated rule (e.g. max recall at FPR ≤ a fixed budget set *before* looking at test results), selected on the validation slice. *(M3: budgets fixed at 0.1 % and 1 % FPR; functions in `src/ids/metrics.py`; undefined metrics reported as null.)*
- Uncertainty: bootstrap CIs on test metrics where cheap.

## 11. Explainability strategy

`shap.TreeExplainer` on the final tree model: global (mean |SHAP|), class-specific (attack class / per attack category), local (per flow: predicted class, probability, top contributing features with sign). Wording rule: "feature X contributed to this prediction", never "X causes the attack". SHAP is also used diagnostically for RQ4 (detecting artefact features).

## 12. Dashboard scope (Milestone 13)

Streamlit, static benchmark demo only, clearly labelled as such: Overview, Detection demo (select a test-set flow → prediction + probability), Explanation (SHAP), Evaluation, Data quality. No live traffic, no simulated "monitoring".

## 13. Reproducibility plan

`uv` venv on Python 3.12 with pinned `requirements.txt`; versioned JSON configs (`configs/feature_manifest_v1.json`, `label_mapping_v1.json`, `experiment_protocol_v1.json`, generated by `scripts/prepare.py`; M3 replaced the planned single YAML); scripts runnable end-to-end; SHA-256 of the downloaded zip and each CSV recorded in `data/README.md`; all result tables generated from saved metrics JSON; data and model artefacts git-ignored.

## 14. Proposed repository structure

Kept deliberately small; folders are created only when the milestone that needs them starts.

```
IDS/
├── data/README.md            # source, download, checksums, expected layout (data itself git-ignored)
├── src/ids/                  # one package: data.py, audit.py, split.py, features.py,
│                             #   train.py, evaluate.py, explain.py
├── scripts/                  # thin CLI entry points (download, audit, train, evaluate, explain)
├── configs/config.yaml
├── tests/
├── dashboard/app.py
├── reports/                  # DATA_AUDIT.md, LEAKAGE_AUDIT.md, EXPERIMENTS.md (+ generated figures)
├── PROJECT_PLAN.md  DATA_CARD.md  MODEL_CARD.md  README.md
├── requirements.txt  .gitignore  LICENSE
```
*Current (M3):* `src/ids/{data,prep,metrics,pipeline}.py`, `scripts/{provenance,audit_raw,convert_parquet,audit_deep,prepare,render_reports,render_m3}.py`, `scripts/hooks/commit-msg`, `configs/*_v1.json`, `reports/{DATA_AUDIT,LEAKAGE_AUDIT,EXPERIMENT_PROTOCOL}.md` + `reports/generated/`.

Deviation from the suggested layout: one `src/ids/` package with modules instead of six sub-packages — the code volume does not justify six packages. `notebooks/` omitted unless exploration needs it.

## 15. Unresolved questions

Status as of Milestone 3 (resolved items kept for the record):

1. ~~Dataset version~~ — resolved M2: improved version (Liu et al.).
2. Licence/terms of the Liu et al. improved dataset — still unresolved (none published). Mitigation: never redistribute; cite.
3. Timestamp timezone — not encoded; consistent with UTC, unverified (M2).
4. ~~Label strings and counts~~ — resolved M2 (DATA_AUDIT §3).
5. ~~Timestamp ordering~~ — resolved M2: files are not time-ordered; all chronological operations sort by (`ts`, `source_file`, `source_row`).
6. Content of Dube (2023) — not read; not used.
7. "BCCC-CIC-IDS-2017" — not investigated, not used.
8. Cause of negative header lengths and protocol-0 rows — unverified; kept and flagged (M3).
9. Rare classes (Heartbleed 3, SQL Injection 4, XSS 5, Infiltration 11, Brute Force 22 test rows) — reported with counts, flagged unstable; no resampling or merging.
10. Reply-direction scan flows labelled BENIGN (M4 hypothesis, MODEL_EVALUATION §8.1) — **decision (2026-09-23): not relabelled in protocol v1**; v1 labels, thresholds, models and results stay unchanged. Any alternative labelling may only be run as a separately documented protocol-v2 sensitivity experiment and never replaces v1 results.

## 16. References (verified via Crossref / publisher / venue pages)

1. I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani. "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." *Proc. 4th ICISSP*, 2018, pp. 108–116. doi:10.5220/0006639801080116
2. G. Engelen, V. Rimmer, W. Joosen. "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study." *2021 IEEE Security and Privacy Workshops (SPW)*, pp. 7–12. doi:10.1109/SPW53761.2021.00009
3. L. Liu, G. Engelen, T. Lynar, D. Essam, W. Joosen. "Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018." *2022 IEEE Conference on Communications and Network Security (CNS)*, pp. 254–262. doi:10.1109/CNS56114.2022.9947235
4. M. Lanvin, P.-F. Gimenez, Y. Han, F. Majorczyk, L. Mé, É. Totel. "Errors in the CICIDS2017 Dataset and the Significant Differences in Detection Performances It Makes." *Risks and Security of Internet and Systems (CRiSIS 2022)*, LNCS, Springer, 2023, pp. 18–33. doi:10.1007/978-3-031-31108-6_2
5. R. Dube. "Faulty use of the CIC-IDS 2017 dataset in information security research." *Journal of Computer Virology and Hacking Techniques*, 2023/2024, pp. 203–211. doi:10.1007/s11416-023-00509-7 *(not yet read)*
6. D. Arp, E. Quiring, F. Pendlebury, A. Warnecke, F. Pierazzi, C. Wressnegger, L. Cavallaro, K. Rieck. "Dos and Don'ts of Machine Learning in Computer Security." *31st USENIX Security Symposium*, 2022. <https://www.usenix.org/conference/usenixsecurity22/presentation/arp>
7. S. M. Lundberg, S.-I. Lee. "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems 30 (NIPS 2017)*.
8. S. M. Lundberg et al. "From local explanations to global understanding with explainable AI for trees." *Nature Machine Intelligence* 2, 56–67, 2020. doi:10.1038/s42256-019-0138-9

Sources pages: CIC dataset page <https://www.unb.ca/cic/datasets/ids-2017.html>; WTMC 2021 tools/dataset page <https://intrusion-detection.distrinet-research.be/WTMC2021/tools_datasets.html>; CNS 2022 pages <https://intrusion-detection.distrinet-research.be/CNS2022/index.html>, `/Dataset_Download.html`, `/CICIDS2017.html`.
