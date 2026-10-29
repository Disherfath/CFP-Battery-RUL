# Frozen paper results

These CSVs contain the reported metrics and derived cell-level errors for *Cell-Level Feature-Preserving Imputation for Battery Remaining Useful Life Prediction*. They cover both datasets and all reported conditions, including negative comparisons.

Run `python reproduction/reproduce_paper.py --output outputs/paper --bootstrap` from the repository root to recreate Tables I-V, Figure 2, and the physical-cell bootstrap checks. See [the data dictionary and reproduction guide](../docs/results.md) for units, aggregation, and scope. `provenance.json` records the original export identifiers and SHA256 hashes.

The data in this directory are available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution belongs to Yifan Li, Yulong Ding, Shuanghua Yang, and Lili Yang for the model-derived results, with the HUST and XJTU dataset attributions given in the guide. This data license is separate from licenses applying to the software. No original voltage/current/capacity curves are included.
