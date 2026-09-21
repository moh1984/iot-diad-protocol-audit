# Release v1.1.0 — revised submission

Accompanies the revised manuscript for IJSSE 45829 (peer review round 1).

Four experiments were rerun or extended, and three previously reported
conclusions changed. See `CHANGELOG.md` for what changed and why.

## Verifying this release

```
pip install -r requirements-lock.txt
python scripts/verify_paper_numbers.py
```

Expected output: `125 values checked, 0 mismatches (119 against the current
manuscript, 6 legacy v1.0 regression checks).`

The script checks the headline values reported in the manuscript against the
files that produced them, and also checks the claims the manuscript makes about
sets of values — that an interval excludes zero, that one interval is separated from
another, that benign recall falls monotonically with depth. It exits non-zero on
any disagreement and trains nothing.

## Reproducing from raw data

The dataset is not redistributed here. Download CIC IoT-DIAD 2024 from the
Canadian Institute for Cybersecurity, set `IOT_DIAD_ROOT`, then run
`scripts/step0_build_sample_by_class.py` followed by `rev1` through `rev4`.
Expect roughly 12 hours in total on four cores; every script is resumable.
