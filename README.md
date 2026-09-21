# Protocol audit on CIC IoT-DIAD 2024

Code and results for *Unexamined Defaults Inflate IoT Intrusion Detection Results:
A Protocol Audit on CIC IoT-DIAD 2024*.

The paper audits four routine decisions in flow-based IoT intrusion detection —
how records are sampled, how they are partitioned, how balancing is compared, and
which hyperparameters are left at library defaults — and measures what each costs.
Every number in the paper can be regenerated from this repository. Experiments that
are reported at two tree depths expose a `MAX_DEPTH` constant at the top of the
script; set it to `None` to regenerate the `*_unbounded.csv` variants.

---

## Dataset

CIC IoT-DIAD 2024, Canadian Institute for Cybersecurity:
<https://www.unb.ca/cic/datasets/iot-diad-2024.html>

> M. Rabbani, J. Gui, F. Nejati, Z. Zhou, A. Kaniyamattam, M. Mirani, G. Piya,
> I. Opushnyev, R. Lu, A. A. Ghorbani. "Device Identification and Anomaly
> Detection in IoT Environments," *IEEE Internet of Things Journal*, vol. 12,
> no. 10, pp. 13625-13643, 2025. doi:10.1109/JIOT.2024.3522863

**We use the flow-based representation only** (`AD_Flow-based-features`,
extracted with CICFlowMeter): 129 CSV files, 19,519,162 flows, eight class
folders. Studies using the packet-based directory report different corpus
dimensions and are not directly comparable.

The dataset is not redistributed here. Download it from the link above and point
the scripts at the directory containing the eight class folders (`Benign`,
`Brute Force`, `DDOS`, `DOS`, `Mirai`, `Recon`, `Spoofing`, `Web-Based`).

---

## Setup

```bash
pip install -r requirements.txt
export IOT_DIAD_ROOT=/path/to/CIC-IoT-DIAD-2024      # Linux / macOS
# setx IOT_DIAD_ROOT "C:\path\to\CIC-IoT-DIAD-2024"  # Windows
```

Scripts read `IOT_DIAD_ROOT` from the environment, falling back to a placeholder
you can edit directly at the top of each file. All outputs are written to
`$IOT_DIAD_ROOT/_working/`.

Results were produced with Python 3.12.3, scikit-learn 1.4.2, pandas 2.2.2 and
numpy 2.2.6 on Windows. `requirements-lock.txt` pins the libraries that affect the
results directly; install from it to reproduce the reported values under the
recorded environment. It is not a full transitive freeze, so exact
bit-for-bit agreement is not guaranteed across platforms.

---

## Running

`scripts/step0_build_sample_by_class.py` must run first — everything downstream
reads its output. It performs a two-pass scan (counting rows before loading them)
and takes 20–60 minutes on the full 9 GB corpus.

| Order | Script | Produces | Paper |
|---|---|---|---|
| 1 | `step0_build_sample_by_class.py` | `iot_diad_sample_by_class.csv`, `true_class_inventory.csv` | Table 1, §5.1 |
| 2 | `exp_A_ports_ablation_depth25.py` | `ablation_ports_depth25.csv` | Table 4, §5.4 |
| 3 | `exp_C2_group_split_multiseed.py` | `split_multiseed_*.csv` | Table 2, §5.2 |
| — | `make_figures.py` | `figures/fig1–3*.pdf` | Figures 1–3 |
| — | `verify_paper_numbers.py` | console report | cross-checks 119 current-manuscript values and claims, plus 6 labelled v1.0 regression checks |
| 4 | `recompute_scored_macro_f1.py` | `scored_macro_f1_*.csv`, `scored_per_family.csv` | Tables 2–3, §5.3 |
| 5 | `exp_B_undersample_control_depth25.py` | `undersample_control_depth25.csv` | Table 5, §5.5 |
| 6 | `exp_F_shap_and_threshold_fixed.py` | `threshold_sweep_*_depth25.csv`, `shap_vs_rf_withports_{depth25,unbounded}.csv` | Table 6, §5.4, §5.6 |
| 7 | `exp_G2_depth_sweep_multiseed.py` | `depth_sweep_multiseed_*.csv` | superseded by `rev4` |

### Revision scripts (v1.1.0)

These supersede the corresponding first-submission experiments and produce every
number in the published tables. Run them after `step0`.

| # | Script | Produces | Paper |
|---|--------|----------|-------|
| R1 | `rev1_protocol_20seeds.py` | `rev1_protocol_20seeds_*.csv`, `rev1_variance_ratio_ci.csv` | Table 2, Figure 2, §5.2 |
| R2 | `rev2_repeated_balancing_and_ablation.py` | `rev2_balancing_*.csv`, `rev2_ablation_*.csv` | Tables 4 and 5, §5.4, §5.5 |
| R3 | `rev3_per_family_20seeds.py` | `rev3_per_family_*.csv` | Table 3, §5.3 |
| R4 | `rev4_depth_sweep_20seeds.py` | `rev4_depth_sweep_20seeds_*.csv` | Table 7, Figure 3, §5.6 |

`rev1` adds a third partitioning protocol in which held-out captures are
accumulated until a fixed proportion of *records* is reached, rather than a fixed
proportion of *files*; this stabilises test-set size while preserving capture
independence. `rev2` adds two control feature pairs to the ablation, without which
a removal result cannot be distinguished from the cost of removing any two
correlated features. Both additions changed a reported conclusion.

Diagnostics (fast, no model training in the first):

- `diag_group_split.py` — which families survive in train vs test under a
  capture-aware split, and why.
- `diag_zero_families.py` — why Recon and Brute Force are never scorable.

Table 6 reports threshold calibration on the depth-25 model only, so only the
depth-25 threshold sweeps are released. The SHAP ranking is reported at both
depths (§5.4), so both variants are included. Set `MAX_DEPTH = None` in the
script to regenerate the unbounded threshold sweeps if needed.

`scripts/common.py` holds shared preprocessing: deduplication, zero-variance
column removal, optional port dropping, and preservation of the `source_file`
group column that makes capture-aware splitting possible.

---

## Sampling protocol

Proportional to true family size, with a floor of 2,000 flows per family, seed 42.
Without the floor, Brute Force (0.02% of the corpus) and Web-Based (0.06%) would
contribute fewer than 100 flows each. **Four families therefore sit above their
true proportions** — Brute Force, Web-Based, Spoofing and Mirai — and results for
those four may be optimistic relative to strictly prevalence-proportional
sampling, and should be interpreted accordingly. Working sample: 125,861 flows,
125,858 after removing three exact duplicates; 72 features after dropping seven
zero-variance columns.

---

## Configuration

All models: random forest, `class_weight="balanced_subsample"`, median imputation,
`max_depth=25` except where depth is the variable under study.

| Experiment | Trees | max_depth | Protocol | Seeds |
|---|---|---|---|---|
| Port ablation | 300 | 25 and None | Random | 15 seeds |
| Protocol comparison | 300 | 25 | All three | 20 seeds |
| Balancing control | 300 | 25 | Random | 15 seeds |
| Threshold calibration | 300 | 25 | Random, 3-way split | 42 |
| Depth sweep | 200 | 6–30, None | All three | 20 seeds |
| SHAP attribution | 300 | 25 and None | Random | 42 |
| Per-family recall | 300 | 25 | Capture-aware | 20 seeds |

This table matches Table 8 of the revised manuscript.

---

## `preliminary/` — read this before reusing anything in it

This directory contains the per-file sampling script we applied to this dataset
**before** auditing it. It holds the script, the ideal-case
file-share calculation, and the actual run summary. It is the worked example analysed in §1 and §5.1, retained
so the distortion can be reproduced.

**It is not a recommended protocol.** Drawing a fixed number of records per
capture file makes a family's sampled share track its share of *files* rather than
its share of flows. On this corpus that overrepresents Mirai roughly 25-fold and
compresses DoS, the genuinely dominant family at 76.09% of flows, to under a third
of its weight. Use `scripts/step0_build_sample_by_class.py` instead.

---

## Verifying the manuscript against these results

```bash
python scripts/verify_paper_numbers.py
```

Checks every reported value against the CSV that produced it, plus the claims the
manuscript makes *about* sets of values — that a confidence interval excludes
zero, that one interval is separated from another, that benign recall falls
monotonically with depth. Of the 125 checks at the time of writing, most validate
values in the current manuscript and a small number are retained as regression
checks against the v1.0 results, so that the two submission versions can still be
compared; the script prints its own count and labels the legacy group. Exits non-zero on any disagreement, runs in under a second, and trains
nothing.

Claim checks matter as much as value checks here. A statement such as "the three
intervals do not overlap" can be false while every interval it refers to is
correctly reported, and no amount of checking the individual numbers would reveal
it.

Freshly regenerated outputs are written to `$IOT_DIAD_ROOT/_working/` to avoid
overwriting the archived reference results in `results/`. To validate a fresh
reproduction with `verify_paper_numbers.py`, first preserve the archived originals,
then copy the regenerated CSV files from `_working/` into `results/` before running
the verifier.

The script guards specifically against the failure mode this paper is about:
results produced under different model configurations being compared as though
they came from one. Every experiment is pinned to `max_depth=25` except the depth
sweep, and the verifier confirms that each value in the manuscript still matches
the archived CSV that produced it, so a configuration change cannot silently
propagate into a table.

## `results/`

Every CSV behind a table or figure in the paper. `*_unbounded.csv` files are the
`max_depth=None` counterparts retained for the two-depth comparisons in §5.4 and
§5.5. Figures in `figures/` are generated by `scripts/make_figures.py`. The PNG outputs are
deterministic: Figures 2 and 3 regenerate byte-identically to those embedded in
the manuscript, and Figure 1 differs only because it was cropped and flattened to
RGB when placed in the journal template. The PDF outputs carry a creation
timestamp and are therefore not byte-reproducible, though the plotted values and
visual content are identical across runs. Figure 1 (share
distortion) comes from `true_class_inventory.csv`; Figure 2 (variance) from
`rev1_protocol_20seeds_raw.csv`, the same run as Table 2; Figure 3 (depth curve)
from `rev4_depth_sweep_20seeds_summary.csv`. Figure numbering matches the paper.

---

## Known limitations

- **Single dataset.** The mechanisms are general; the magnitudes are specific to
  this corpus's file organisation.
- **Labels are folder-derived.** The dataset's own `Label` column contains the
  placeholder `NeedManualLabel` for every record, so benign traffic captured
  during an attack window is labelled as attack.
- **Two families cannot be evaluated under capture-aware splitting at all.** Recon
  and Brute Force each occupy one capture file. No seed or repetition count
  resolves this.
- **Benign metrics under capture-aware splitting rest on a subset of draws.**
  Benign occupies four captures, so it is testable in 11 of 20 draws under
  file-proportion splitting and 9 of 20 under size-matched splitting. Those counts
  are reported wherever benign figures appear.
- **Uncalibrated probabilities.** Thresholds are applied to raw random-forest
  scores.

---

## Citing

```bibtex
@misc{alkhazaleh2026audit,
  title   = {Unexamined Defaults Inflate IoT Intrusion Detection Results:
             A Protocol Audit on CIC IoT-DIAD 2024},
  author  = {Alkhazaleh, Mohammad and Baklizi, Mahmoud and
             Mjlae, Salameh A. and Alzghoul, Musab and Al~Sardy, Loui},
  year    = {2026},
  note    = {Manuscript under review, International Journal of Safety and
             Security Engineering}
}
```

## License

Code released under the MIT License (see `LICENSE`). The dataset is subject to the
terms of the Canadian Institute for Cybersecurity.
