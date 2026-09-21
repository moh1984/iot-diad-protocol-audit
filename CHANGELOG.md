# Changelog

## v1.1.0 — 2026-09-16

Accompanies the revised manuscript submitted to the *International Journal of
Safety and Security Engineering* (45829). Four experiments were rerun or extended
after peer review, and three previously reported conclusions changed as a result.
Those changes are listed first, because they are the substantive content of this
release.

### Conclusions that changed

**The port features are not redundant.** The first submission reported that
removing `Src Port` and `Dst Port` left performance essentially unchanged, and
concluded they were redundant. That comparison had no control: removing any two of
72 correlated features costs something, and without a baseline the port result
could not be interpreted. Adding two control pairs reversed the finding. At
`max_depth=25`, removing the port pair costs 0.0250 macro-F1 (95% CI
0.0239–0.0261) on the eight-class task, against 0.0027 for the pair ranked first
and second by SHAP and 0.0006 for an arbitrary pair; the port-pair interval is
separated from both controls. The underlying claim — that attribution rank does
not establish necessity — survives, but in the opposite direction from the one
originally argued.

**The variance ratio has no defensible point estimate.** The first submission
reported capture-aware accuracy as "roughly sixty-seven times noisier", a ratio of
two standard deviations each estimated from five draws. At twenty draws the
bootstrap 95% interval is 44.9–125.9. The point estimate has been removed from the
manuscript and intervals are reported throughout.

**Benign-class results no longer rest on two draws.** At five seeds, benign
traffic was testable in only two capture-aware draws, giving a per-family recall
of 0.2321. At twenty seeds it is testable in eleven, and the estimate is 0.2872.

### New in this release

- `scripts/rev1_protocol_20seeds.py` — protocol comparison at twenty seeds, and a
  third partitioning protocol in which captures are accumulated until a fixed
  proportion of *records* is held out rather than a fixed proportion of *files*.
  Test-partition size under the original protocol ranges from 8.0% to 40.3% of
  records across twenty draws; the size-matched protocol constrains it to
  20.0%–28.5%. Fixing test size reduces the multiclass accuracy standard deviation
  by about a quarter, so variability in test size accounts for part of the excess
  variance but not most of it; the two sources cannot be separated completely,
  since the size-matched protocol also changes which captures are selected.
- `scripts/rev2_repeated_balancing_and_ablation.py` — balancing and feature
  ablation repeated over fifteen seeds with paired per-seed differences, plus the
  two control feature pairs described above.
- `scripts/rev3_per_family_20seeds.py` — per-family recall at twenty seeds under
  both capture-aware protocols, recording which families were scorable in each
  draw.
- `scripts/rev4_depth_sweep_20seeds.py` — depth sweep at twenty seeds across all
  three protocols, replacing the five-seed sweep.

### Changed

- `scripts/verify_paper_numbers.py` extended from 55 to 125 checks (119 against
  the current manuscript and 6 retained as labelled v1.0 regression checks),
  retargeted at the revision result files, and given claim-level checks: that a confidence
  interval excludes zero, that one interval is separated from another, that
  benign recall falls monotonically with depth, and that the macro-F1 ordering is
  protocol-dependent. These matter because a statement such as "the three
  intervals do not overlap" can be false while every interval it refers to is
  correctly reported.
- `scripts/make_figures.py` now builds Figure 2 from `rev1` and Figure 3 from
  `rev4`. The PNG outputs are deterministic and Figures 2 and 3 regenerate
  byte-identically to those embedded in the manuscript; the PDF outputs carry a
  creation timestamp and are not byte-reproducible, though their content is.
- `CITATION.cff` version and release date.

### Retained

Results from the first submission are kept in `results/` under their original
names, so the two versions can be compared. Where a revision file supersedes an
earlier one, both are present; the paper cites the revision figures.

---

## v1.0.1 — 2026-08-23

Line-ending normalisation (`.gitattributes`), corrected bibliographic details for
the dataset reference, and repository hygiene. No change to any result.

## v1.0.0 — 2026-08-23

Initial release accompanying the first submission.
