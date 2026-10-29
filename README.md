# CFP-Battery-RUL

Code and reproducible result exports for **Cell-Level Feature-Preserving Imputation for Battery Remaining Useful Life Prediction**, accepted at the first workshop of ICCBD+AI 2026.

**Authors:** Yifan Li, Yulong Ding, Shuanghua Yang, and Lili Yang.  
**Repository maintainer:** Yifan Li ([Disherfath](https://github.com/Disherfath)).

CFP trains a time-series imputer to preserve the 18 cell-level descriptors consumed by a battery lifetime regressor. The feature objective combines information from the first 50 cycles while the SAITS backbone operates within each cycle. It is used during training and adds no inference-network parameters. The release includes matched Base, CFP, Direction, and Joint arms, along with the reference methods used in the paper.

## Contents

| Location | Purpose |
| --- | --- |
| `src/batt_rul/` | Model adapters, feature losses, data preparation, missingness, regression heads, and evaluation |
| `scripts/` | Upstream setup, cache preparation, and train/evaluate entry points |
| `configs/` | Frozen physical-cell roles, paper protocol, and numerical source provenance |
| `results/` | Archived paper metrics and derived cell-level errors |
| `reproduction/` | Reconstruction of Tables I-V, Figure 2, and paired bootstrap results |
| `tests/` | Numerical and protocol checks |

The complete result dictionary and interpretation are in [docs/results.md](docs/results.md). Both datasets and all reported missingness conditions are included.

## Installation

Use Python 3.10 or newer and Git. Install a PyTorch build appropriate for the machine's CUDA version before running GPU experiments.

```bash
git clone https://github.com/Disherfath/CFP-Battery-RUL.git
cd CFP-Battery-RUL
python -m pip install -e ".[dev]"
```

The release checks were run with Python 3.12.3, PyTorch 2.3.0+cu121, NumPy 2.5.2, pandas 2.3.3, scikit-learn 1.9.0, SciPy 1.18.0, and Matplotlib 3.9.0. Dependencies in `pyproject.toml` allow a wider range; numerical results can vary with package versions, hardware, and floating-point kernels.

## Reconstruct the reported tables and figure

This route uses the included frozen result files and requires no raw battery curves or neural retraining:

```bash
python reproduction/reproduce_paper.py --output outputs/paper --bootstrap
```

The program verifies the result-file hashes, recomputes the repeated-run summaries, writes LaTeX tables, regenerates Figure 2, and replays the physical-cell bootstrap intervals. Choose a new output directory for each invocation. See [the reproduction guide](docs/results.md) for aggregation rules and file definitions.

## Prepare the upstream models

The setup command downloads and verifies the fixed author revisions used by the adapters. Run it for each backbone needed:

```bash
python scripts/setup_backbones.py saits
python scripts/setup_backbones.py brits
python scripts/setup_backbones.py t1
```

The default locations are `third_party/SAITS`, `third_party/BRITS`, and `third_party/T1`. These downloads retain their original MIT licenses and are excluded from Git. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records upstream authors, commits, licenses, and the scope of our adaptations.

## Prepare the battery datasets

Download the original releases from their owners:

| Dataset | Official release | Train / validation / test cells | Frozen roles |
| --- | --- | --- | --- |
| HUST | [Mendeley Data v2](https://doi.org/10.17632/nsc7hnsg4s.2) | 47 / 8 / 22 | `configs/hust_roles.csv` |
| XJTU | [Zenodo record 10963339](https://doi.org/10.5281/zenodo.10963339) | 29 / 7 / 19 | `configs/xjtu_roles.csv` |

Both releases use CC BY 4.0. Raw archives and measured curves are obtained separately; they are not bundled here. Extract the files so that the preparation roots have these structures:

```text
data/raw/hust/
    1-1.pkl
    ...
data/raw/xjtu/
    Batch-1/
        ...battery-....mat
    ...
    Batch-6/
```

Pass the directory containing the HUST pickle files and the directory containing the six XJTU `Batch-*` folders:

```bash
python scripts/build_hust_cache.py data/raw/hust data/cache/hust
python scripts/build_xjtu_cache.py data/raw/xjtu data/cache/xjtu
```

Each cache stores the first 50 usable cycles, resampled to 128 positions, with voltage, current, and capacity channels. HUST uses usable recorded-cycle count for its endpoint, including the documented two-cycle prefix exclusion for cell `7-5`. XJTU uses the released `cycle_life`. The target is remaining cycles after the 50-cycle history. The source HUST study used an 80%-of-nominal-capacity endpoint; this preparation treats the released histories as endpoint-observed and does not perform right-censoring inference.

## Train, select the regression head, and evaluate

Training and regression-head selection use the fixed training and validation roles. Test evaluation is a separate command that loads the frozen model and selected heads. Use a fresh output directory for every run.

Example: HUST CFP-SAITS with the first paper initialization:

```bash
python scripts/run_experiment.py train --dataset hust --arm cfp --seed 20261005 --cache data/cache/hust --splits configs/hust_roles.csv --author-root third_party/SAITS --output runs/hust_cfp_s20261005 --device cuda:0
python scripts/run_experiment.py evaluate --run runs/hust_cfp_s20261005 --cache data/cache/hust --splits configs/hust_roles.csv --author-root third_party/SAITS --output outputs/hust_cfp_s20261005_test --device cuda:0
```

Example: the corresponding XJTU run:

```bash
python scripts/run_experiment.py train --dataset xjtu --arm cfp --seed 20261005 --cache data/cache/xjtu --splits configs/xjtu_roles.csv --author-root third_party/SAITS --output runs/xjtu_cfp_s20261005 --device cuda:0
python scripts/run_experiment.py evaluate --run runs/xjtu_cfp_s20261005 --cache data/cache/xjtu --splits configs/xjtu_roles.csv --author-root third_party/SAITS --output outputs/xjtu_cfp_s20261005_test --device cuda:0
```

Supported `--arm` values and author roots:

| Arms | Author root |
| --- | --- |
| `base`, `cfp`, `direction`, `joint`, `saits64` | `third_party/SAITS` |
| `t1` | `third_party/T1` |
| `brits-base`, `brits-masked`, `brits-cfp`, `brits-direction` | `third_party/BRITS` |
| `linear`, `c0`, `observed-only` | Unused; these arms fit the regression heads without neural training |

The paper settings are recorded in [configs/paper_protocol.json](configs/paper_protocol.json): 800 neural updates, learning rate 0.001, feature weight 0.1, and the original head-search grids. SAITS arms use initializations `20261005`, `20261018`, and `20261019`; T1 uses `20261008`, `20261018`, and `20261019`; BRITS uses `20261005`. Repeat the commands with the corresponding seeds and distinct output directories to cover the paper's runs. The Joint arm defaults to its selected lifetime-loss weight, 0.3 on HUST and 0.1 on XJTU.

The conditions are clean, random time-point missingness, contiguous block missingness, and observation-window missingness. Each missing condition hides 51/128 positions (39.84%), with three fixed mask repetitions. The three initializations share the physical-cell split and missingness protocol. Whole-cell SAITS processes 50 cycles per update; SAITS-64 samples 64, so the shared update budget gives different sample exposure.

Training outputs include configuration and provenance records, the model state, training history, completed-feature banks, the validation search grid, and frozen regression heads. Evaluation produces `predictions.csv`, `metrics.json`, and a receipt identifying the test cells and training run.

## Engineering checks and reproduction status

```bash
python -m pytest -q
```

Tests requiring upstream models are skipped until their fixed sources are installed. A shortened run can be requested with `--updates 2 --smoke`; evaluating that run also requires `--allow-smoke`. Smoke outputs are marked separately from paper experiments.

The reported 800-update values in `results/` come from the original study. Release validation on 29 October 2026 passed 61 tests, saved-result verification for 14 method/dataset groups, and all 24 paired bootstrap comparisons. CUDA smoke runs completed training, validation-grid selection, and independent frozen testing for CFP-SAITS on HUST and Joint-SAITS on XJTU (two updates each), plus T1, BRITS-CFP, and SAITS-64 on HUST (one update each). The first-update loss and gradient norm for CFP-SAITS, T1, BRITS-CFP, and SAITS-64 exactly matched their archived original-run histories. Full-budget reruns of the packaged release have not been performed.

## Citation and licensing

Please cite the paper when using CFP or its experimental results. [CITATION.cff](CITATION.cff) provides the complete author list and paper metadata. The accepted paper's proceedings DOI and page numbers will be added when available.

- Original project software: [MIT](LICENSE).
- Upstream software: original licenses and notices in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `LICENSES/`.
- Derived result data: [CC BY 4.0](results/README.md), with HUST and XJTU source attribution.

Personal-account hosting identifies the maintainer; it does not replace the paper authorship or upstream attribution.
