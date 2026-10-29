"""Regenerate paper tables and Figure 2 from frozen derived results.

This program reads the reported error exports, verifies the three-initialization
main table from physical-cell errors, and optionally replays the cell bootstrap.
Model fitting is handled by the training/evaluation entry points in scripts/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {"mcar_time40": "Random", "block40": "Block", "window40": "Window"}
SEEDS = {20261005, 20261018, 20261019}
TOL = 1e-12


def read(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / name, dtype={"cell_id": str}, float_precision="round_trip")


def one(frame: pd.DataFrame, **filters):
    for key, value in filters.items():
        frame = frame.loc[frame[key].eq(value)]
    if len(frame) != 1:
        raise ValueError(f"Expected one row for {filters}, got {len(frame)}")
    return frame.iloc[0]


def number(value, digits=2):
    return "--" if pd.isna(value) else f"{float(value):.{digits}f}"


def mean_sd(mean, sd, digits=2, bold=False):
    value = number(mean, digits)
    if not pd.isna(sd):
        value += rf"\pm {float(sd):.{digits}f}"
    return "$" + (rf"\mathbf{{{value}}}" if bold else value) + "$"


def tex_table(output, name, columns, header, rows, caption, wide=False):
    environment = "table*" if wide else "table"
    lines = [rf"\begin{{{environment}}}[!t]", rf"\caption{{{caption}}}",
             r"\centering\small\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
             header + r"\\", r"\midrule"]
    lines += [" & ".join(row) + r"\\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{environment}}}"]
    (output / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_hashes(root):
    manifest = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = root / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Frozen result changed: {record['file']}")
    return len(manifest["files"])


def verify_table2(root, table):
    saits = read(root, "cell_errors_saits.csv")
    joint = read(root, "cell_errors_joint.csv")
    references = read(root, "cell_errors_references.csv")
    verified = []
    for _, row in table.iterrows():
        if row["method"] in ("SAITS Base", "CFP-SAITS", "Direction"):
            cells = saits.loc[saits.arm.eq(row.arm) & saits.family.eq("ridge")]
        elif row["method"].startswith("Joint"):
            cells = joint.loc[joint.family.eq(row.family)]
        else:
            kind = {"SAITS-64": "saits64", "T1-within": "t1_within"}[row["method"]]
            cells = references.loc[references.kind.eq(kind) & references.family.eq("ridge")]
        for dataset, count in (("hust", 22), ("xjtu", 19)):
            subset = cells.loc[cells.dataset.eq(dataset)]
            expected_seeds = {20261008, 20261018, 20261019} if row["method"] == "T1-within" else SEEDS
            if set(subset.seed) != expected_seeds or subset.duplicated(["seed", "cell_id"]).any():
                raise ValueError(f"Unexpected repeat coverage for {row['method']} {dataset}")
            if not subset.groupby("seed").cell_id.nunique().eq(count).all():
                raise ValueError("Physical-cell coverage mismatch")
            macro = subset.groupby("seed")[["ape", "ae", "mse"]].mean()
            macro["mape"] = macro.ape * 100
            macro["mae"] = macro.ae
            macro["rmse"] = np.sqrt(macro.mse)
            for metric in ("mape", "mae", "rmse"):
                np.testing.assert_allclose(
                    [macro[metric].mean(), macro[metric].std(ddof=1)],
                    [row[f"{dataset}_{metric}_mean"], row[f"{dataset}_{metric}_sd"]],
                    rtol=1e-11, atol=1e-10,
                )
            verified.append({"dataset": dataset, "method": row["method"], "cells": count})
    return verified


def export_tables(root, output):
    single = read(root, "table1_single_initialization.csv")
    rows = []
    for _, r in single.iterrows():
        if r.label == "Joint SAITS: MLP":
            rows.append([r.label, rf"\multicolumn{{2}}{{c}}{{{r.hust_ridge:.2f}}}",
                         rf"\multicolumn{{2}}{{c}}{{{r.xjtu_ridge:.2f}}}"])
        else:
            rows.append([r.label] + [number(r[k]) for k in
                         ("hust_ridge", "hust_rbf", "xjtu_ridge", "xjtu_rbf")])
    tex_table(output, "table1", "lrrrr",
              r" & \multicolumn{2}{c}{HUST} & \multicolumn{2}{c}{XJTU}\\"
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}Configuration & Ridge & RBF & Ridge & RBF",
              rows, r"Single-initialization test MAPE (\%). Joint MLP spans the regression-family columns.")

    repeated = read(root, "table2_three_initializations.csv")
    checks = verify_table2(root, repeated)
    rows = []
    for _, r in repeated.iterrows():
        values = [r["method"]]
        for dataset in ("hust", "xjtu"):
            for metric in ("mape", "mae", "rmse"):
                key = f"{dataset}_{metric}"
                best = np.isclose(r[key + "_mean"], repeated[key + "_mean"].min(), rtol=0, atol=1e-12)
                values.append(mean_sd(r[key + "_mean"], r[key + "_sd"], bold=best))
        rows.append(values)
    tex_table(output, "table2", "lrrrrrr",
              r" & \multicolumn{3}{c}{HUST} & \multicolumn{3}{c}{XJTU}\\"
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}Method & MAPE & MAE & RMSE & MAPE & MAE & RMSE",
              rows, r"Three-initialization test results (mean $\pm$ sample SD). MAPE is in percent; MAE/RMSE are in cycles.", True)

    conditions = read(root, "table3_conditions_core.csv").merge(
        read(root, "table3_conditions_references.csv"), on=["dataset", "condition"], validate="one_to_one")
    if len(conditions) != 6:
        raise ValueError("Expected all six dataset-pattern rows")
    conditions.to_csv(output / "table3_combined.csv", index=False)
    rows = []
    for _, r in conditions.iterrows():
        rows.append([r.dataset.upper(), PATTERNS[r.condition],
                     mean_sd(r.base_mean, r.base_sd), mean_sd(r.cfp_mean, r.cfp_sd),
                     number(r.li), number(r.observed_only), mean_sd(r.joint_mlp_mean, r.joint_mlp_sd),
                     mean_sd(r.saits64_mape_pct_mean, r.saits64_mape_pct_sample_sd),
                     mean_sd(r.t1_within_mape_pct_mean, r.t1_within_mape_pct_sample_sd)])
    tex_table(output, "table3", "llrrrrrrr",
              "Dataset & Pattern & Base & CFP-SAITS & LI & Observed-only & Joint MLP & SAITS-64 & T1-within",
              rows, r"Test MAPE (\%) by missingness pattern. Neural results are mean $\pm$ sample SD over three initializations.", True)

    features = read(root, "table4_hust_features.csv")
    rows = [[r["method"], mean_sd(r.point_mean, r.point_sd, 3),
             mean_sd(r.feature_mean, r.feature_sd, 3), mean_sd(r.mape_mean, r.mape_sd)]
            for _, r in features.iterrows()]
    tex_table(output, "table4", "lrrr", "Method & Point MAE & Feature MSE & MAPE", rows,
              r"HUST reconstruction and descriptor retention. Neural results use three initializations; LI is deterministic.")

    sensitivity = read(root, "validation_weight_sensitivity.csv")
    selected = sensitivity.loc[sensitivity.analysis.eq("same_grid_selected") & sensitivity.family.eq("ridge")]
    rows, records = [], []
    for weight in (0.0, .01, .03, .1, .3):
        record = {"weight": weight}
        for dataset in ("hust", "xjtu"):
            if weight == 0:
                r = one(read(root, f"validation_reference_grid_{dataset}.csv"),
                        completion="saits_whole800_base", family="ridge")
                record[dataset] = float(r.missing_mape_pct)
            else:
                record[dataset] = float(one(selected, dataset=dataset, weight=weight).mape_pct)
        rows.append(["Base (no CFP)" if weight == 0 else str(weight), number(record["hust"]), number(record["xjtu"])])
        records.append(record)
    pd.DataFrame(records).to_csv(output / "table5_validation_ridge.csv", index=False)
    tex_table(output, "table5", "lrr", "Feature weight & HUST & XJTU", rows,
              r"Validation MAPE (\%) with Ridge. Each weight uses the same downstream search grid; the main weight remains 0.1.")
    return checks


def verify_pairing(root, replay_bootstrap=False):
    cells = read(root, "paired_cells.csv")
    summary = read(root, "paired_summary.csv")
    by_init = read(root, "paired_by_initialization.csv")
    rng = np.random.default_rng(20261018)
    checked = 0
    for dataset, n_cells in (("hust", 22), ("xjtu", 19)):
        for condition in ("missing3", "mcar_time40", "block40", "window40"):
            for comparator in ("Base", "LI", "OO"):
                subset = cells.loc[cells.dataset.eq(dataset) & cells.condition.eq(condition)
                                   & cells.comparator.eq(comparator)]
                saved = one(summary, dataset=dataset, condition=condition, comparator=comparator)
                runs = by_init.loc[by_init.dataset.eq(dataset) & by_init.condition.eq(condition)
                                   & by_init.comparator.eq(comparator)]
                if len(subset) != n_cells or subset.cell_id.nunique() != n_cells or set(runs.seed) != SEEDS:
                    raise ValueError("Paired result coverage mismatch")
                delta = subset.delta_pp.to_numpy()
                np.testing.assert_allclose(delta, subset.cfp_mean_ape_percent - subset.comparator_mean_ape_percent,
                                           rtol=0, atol=1e-10)
                np.testing.assert_allclose([delta.mean(), np.median(delta), runs.mean_delta_pp.std(ddof=1)],
                                           [saved.mean_delta_pp, saved.median_delta_pp, saved.seed_delta_sd_pp],
                                           rtol=1e-10, atol=1e-10)
                if int((delta < -TOL).sum()) != int(saved.improved_cells):
                    raise ValueError("Improved-cell count differs")
                if replay_bootstrap:
                    draws = delta[rng.integers(0, n_cells, size=(20000, n_cells))].mean(axis=1)
                    ci = np.quantile(draws, [.025, .975], method="linear")
                    np.testing.assert_allclose(ci, [saved.cell_bootstrap95_low_pp, saved.cell_bootstrap95_high_pp],
                                               rtol=0, atol=1e-9)
                checked += 1
    if checked != 24:
        raise ValueError("Missing paired comparisons")
    return checked


def plot_figure2(root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = read(root, "paired_cells.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9,
                         "axes.titlesize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.14, 2.82), constrained_layout=True)
    for ax, dataset, letter in zip(axes, ("hust", "xjtu"), ("a", "b")):
        p = cells.loc[cells.dataset.eq(dataset) & cells.condition.eq("missing3") & cells.comparator.eq("Base")]
        x, y = p.comparator_mean_ape_percent.to_numpy(), p.cfp_mean_ape_percent.to_numpy()
        upper = np.ceil(max(x.max(), y.max()) * 1.05 / 5) * 5
        improved = y < x - TOL
        ax.plot([0, upper], [0, upper], color="#82909E", lw=1, ls="--", zorder=1)
        for mask, color, marker in ((improved, "#237A68", "o"), (~improved, "#B56A30", "^")):
            ax.scatter(x[mask], y[mask], s=26, c=color, marker=marker, edgecolor="white", linewidth=.45, zorder=3)
        ax.set(xlim=(0, upper), ylim=(0, upper), xlabel="Base SAITS mean cell APE (%)",
               ylabel="CFP-SAITS mean cell APE (%)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#E3E8EC", lw=.5, zorder=0)
        ax.set_title(f"({letter}) {dataset.upper()} ({len(p)} cells)", loc="left", fontweight="bold", pad=7)
        ax.text(.05, .95, f"{improved.sum()}/{len(p)} cells improved", transform=ax.transAxes, va="top",
                fontsize=8.5, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1})
    fig.savefig(output / "figure2_paired_cells.pdf")
    fig.savefig(output / "figure2_paired_cells.png", dpi=350)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "paper")
    parser.add_argument("--bootstrap", action="store_true", help="Replay all 24 physical-cell bootstrap intervals")
    parser.add_argument("--tables-only", action="store_true", help="Skip Figure 2 rendering")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory to preserve previous outputs")
    n_files = verify_hashes(args.results)
    args.output.mkdir(parents=True)
    checks = export_tables(args.results, args.output)
    paired = verify_pairing(args.results, args.bootstrap)
    if not args.tables_only:
        plot_figure2(args.results, args.output)
    report = {"verified_input_files": n_files, "main_table_groups": checks,
              "paired_comparisons": paired, "bootstrap_replayed": args.bootstrap,
              "bootstrap_draws": 20000 if args.bootstrap else 0,
              "source": "frozen derived errors from the reported runs", "new_model_training": False}
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
