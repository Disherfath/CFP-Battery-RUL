# Third-party code and data notices

The original CFP battery-task software is distributed under the root MIT license. Third-party components retain their own copyright notices and license terms. Full upstream license texts are preserved in `LICENSES/`; the root software license does not relicense the battery datasets or derived experimental data.

## SAITS

- Repository: https://github.com/WenjieDu/SAITS
- Pinned commit: `660b87f19c1277065f314f24134f646229e89ca9`
- Copyright: Copyright 2022 Wenjie Du.
- License: MIT; see [SAITS-MIT.txt](LICENSES/SAITS-MIT.txt).
- Upstream files used: `modeling/saits.py`, `modeling/layers.py`, and `modeling/utils.py`.
- Integration: `src/batt_rul/saits_task_adapter.py` loads the author model and native reconstruction losses. The battery input adapter, whole-cell feature objective, and experiment runners are project-specific additions.
- Paper: Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 2023. https://doi.org/10.1016/j.eswa.2023.119619

Preserve upstream file headers and the accompanying license when downloading or redistributing the author source.

## BRITS

- Repository: https://github.com/caow13/BRITS
- Pinned commit: `fc0a3a472a6d99a6934e471d40c25ed0b029b501`
- Copyright: Copyright (c) 2018 Dawn90.
- License: MIT; see [BRITS-MIT.txt](LICENSES/BRITS-MIT.txt).
- Integration: `src/batt_rul/brits_cycle_imputer.py` adapts the author RITS/BRITS implementation to three channels and 128 positions, with device-safe state creation and a battery-specific input boundary. The file preserves the original MIT notice and documents the adaptation.
- Paper: Wei Cao, Dong Wang, Jian Li, Hao Zhou, Lei Li, and Yitan Li. BRITS: Bidirectional Recurrent Imputation for Time Series. NeurIPS, 2018. https://papers.neurips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html

The BRITS backbone belongs to its original authors; CFP and the downstream battery protocol are the contributions of this project.

## T1

- Repository: https://github.com/Oppenheimerdinger/T1
- Pinned commit: `823b76aab57a2e60eb5029caab06fdfec8b19256`
- Copyright: Copyright (c) 2025 Dongik Park, Hyunwoo Ryu, Suahn Bae, Keondo Park, Hyung-Sin Kim.
- License: MIT; see [T1-MIT.txt](LICENSES/T1-MIT.txt).
- Upstream file used: `models/T1.py`.
- Integration: `src/batt_rul/t1_cycle_adapter.py` verifies and loads the author core. `src/batt_rul/t1_within_cycle_adapter.py` treats each battery cycle as a 128-position, three-variable V/I/Q sequence.
- Paper: Dongik Park, Hyunwoo Ryu, Suahn Bae, Keondo Park, and Hyung-Sin Kim. T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation. ICLR, 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/41ff2b921a875902fad0c1840593c19a-Abstract-Conference.html

This repository reports a battery-task adaptation of T1, with its data preparation and evaluation settings documented separately from the original paper.

## BatteryML

- Repository: https://github.com/microsoft/BatteryML
- Pinned commit used for HUST split membership: `2861ae3b8c79938c7fc8e6fe9986b799ca71c7dd`.
- Copyright: Copyright (c) Microsoft Corporation.
- License: MIT; see [BatteryML-MIT.txt](LICENSES/BatteryML-MIT.txt).
- Reuse: the HUST training/test membership and commented 47/8/22 training/validation/test split are retained in `src/batt_rul/data/hust.py`.
- Paper: Han Zhang et al. BatteryML: An Open-source Platform for Machine Learning on Battery Degradation. ICLR, 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/4048c740bec4c682b17c640f584d2762-Abstract-Conference.html

## Battery datasets and derived results

The upstream data licenses are separate from the software licenses. Raw battery curves are not included in this repository; obtain them from the official records.

### HUST

- Dataset: The Dataset for: Real-time personalized health status prediction of lithium-ion batteries using deep transfer learning.
- Creators: Ye Yuan, Guijun Ma, and Songpei Xu.
- Release: version 2, 24 May 2022.
- Official data and license record: https://data.mendeley.com/datasets/nsc7hnsg4s/2
- Dataset DOI: https://doi.org/10.17632/nsc7hnsg4s.2
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- Related publication: https://doi.org/10.1039/D2EE01676A

### XJTU

- Dataset: Project - Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis.
- Creator: Fujin Wang, Xi'an Jiaotong University.
- Release: version V1, 12 April 2024.
- Official data and license record: https://zenodo.org/records/10963339
- Dataset DOI: https://doi.org/10.5281/zenodo.10963339
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/), verified in the official record's `metadata.license.id` field.
- Related publication: https://doi.org/10.1038/s41467-024-48779-z

The published results are derived by this project's preprocessing, missingness simulation, model training, and evaluation. Model predictions are generated by this project and are not predictions supplied by the original dataset authors. Source cell identifiers and relevant lifetime targets retain dataset attribution. See `docs/results.md` and the notices accompanying `results/` for the released result files and their CC BY 4.0 terms.

When reusing derived data, retain the upstream dataset citations, identify this project's transformations, and acknowledge the CFP paper. The MIT software license does not remove the datasets' attribution requirements.

## Other dependencies

Separately installed packages, including PyTorch, NumPy, pandas, SciPy, scikit-learn, and Matplotlib, retain their respective licenses. These package distributions are not included in the repository.

Upstream licenses and dataset records were checked on 29 October 2026.
