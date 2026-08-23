# Preliminary analysis — the audited example, NOT a recommended protocol

`step1_test_merge.py` draws a fixed number of records (1,000) from **every CSV
file** in the corpus. This is the sampling procedure audited in §5.1 of the paper.

## What the actual run produced

`build_summary.csv` is the unmodified output of this script on the full corpus.
It is the source of the figures cited in §1 and §5.1:

| Family | Files | True rows | Sampled rows | **Observed share** | Share of files | True share |
|---|---:|---:|---:|---:|---:|---:|
| DDoS | 61 | 3,478,814 | 61,000 | **49.19%** | 47.29% | 17.82% |
| Mirai | 29 | 174,588 | 29,000 | **23.39%** | 22.48% | 0.89% |
| DoS | 27 | 14,853,087 | 22,000 | **17.74%** | 20.93% | **76.09%** |
| Benign | 4 | 398,330 | 4,000 | **3.23%** | 3.10% | 2.04% |
| Spoofing | 3 | 157,238 | 3,000 | 2.42% | 2.33% | 0.81% |
| Web-Based | 3 | 11,328 | 3,000 | 2.42% | 2.33% | 0.06% |
| Brute Force | 1 | 3,619 | 1,000 | 0.81% | 0.78% | 0.02% |
| Recon | 1 | 442,158 | 1,000 | 0.81% | 0.78% | 2.27% |

Total: 124,000 records.

## Why it is wrong

The observed share tracks the file-share column, not the true-share column. DoS
holds 76.09% of all flows and receives 17.74% of the sample; Mirai holds 0.89%
and receives 23.39%, an overrepresentation of roughly 26-fold. The two families
differ in size by a factor of 85 and in file count by a factor of 1.07.

The mechanism: Benign concentrates roughly 100,000 flows into each of four files,
while Mirai spreads 174,588 flows across twenty-nine. A quota of *n* records per
file rewards fragmentation and penalises concentration.

**Why the observed shares are close to but not equal to the file shares.** Under
an ideal uniform quota that every capture could satisfy, each family would
contribute exactly 1,000 x (its file count) and the sampled share would equal the
file share. In practice five of the 27 DoS captures hold fewer than 1,000 flows
and contribute all of their rows instead, so DoS receives 22,000 rather than
27,000 and every other family's share is inflated slightly in compensation. This
is also why 124 of the 129 captures are represented in the corrected working
sample (see §4.2 of the paper).

`theoretical_file_share_expectation.csv` tabulates that ideal case for comparison.
It is a calculation derived from `../results/true_class_inventory.csv`, not a run
output.

## What to use instead

`../scripts/step0_build_sample_by_class.py` — proportional to true family size
with a documented floor for small families.
