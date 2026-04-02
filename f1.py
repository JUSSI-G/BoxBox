"""
F1 Championship Simulator  v4.0
================================
Bachelor's Thesis Tool — LUT University
"The impact of software engineering on strategy development in Formula One"

WHAT THIS FILE DOES:
    Uses the trained Random Forest model from rf.py to answer the
    counterfactual question:

        "If every driver had used perfect pit strategy, would the
         championship winner have changed?"

    No hardcoded correction factors. No assumed field spread values.
    The simulation is entirely driven by what the RF learned from real data.

HOW IT WORKS:
    1. Load f1_rf_dataset.csv (produced by analyser.py + rf.py)
    2. Train the RF model identically to rf.py
    3. For each driver-race, create a "perfect strategy" copy by zeroing
       out window_penalty_s, exec_penalty_s, and avg_window_error_laps
       — keeping grid_position and era_code unchanged (car quality is fixed)
    4. Run both actual and perfect-strategy features through the RF
       to get predicted points under each scenario
    5. Sum predicted points per driver per season → championship standings
    6. Compare: does the predicted champion change?

WHY THIS IS VALID:
    The RF was trained on real historical data where variance and points
    are both observed. Zeroing the variance features is a standard
    counterfactual prediction — asking the model to extrapolate to a
    scenario where strategy errors were eliminated. The RF's own learned
    weights determine how much those features matter; there are no
    manually chosen correction coefficients.

    Limitation acknowledged: zeroing variance features is mild
    extrapolation below the minimum observed values. This is noted
    as a limitation in the thesis methodology section.

RELATIONSHIP TO OTHER FILES:
    analyser.py  →  computes variance metrics from raw pitstop data
    rf.py        →  trains RF, saves f1_rf_dataset.csv, returns model
    f1.py (this) →  loads dataset + model, runs championship simulation

Usage:
    python analyser.py          # produces f1_rf_dataset.csv via rf.py
    python f1.py                # simulate all years in dataset
    python f1.py --year 2010    # single season
    python f1.py --start 1994 --end 2024
    python f1.py --no-plot
"""

import pandas as pd
import numpy as np
import argparse, os, sys
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# ── Path resolution ────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(_HERE, "outputs")
RF_DATASET_CSV = os.path.join(OUTPUTS_DIR, "f1_rf_dataset.csv")

# Ensure sibling files are importable regardless of cwd
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Must match rf.py exactly — the model is trained on these columns
FEATURE_COLS = [
    "grid_position",
    "avg_window_error_laps",
    "window_penalty_s",
    "exec_penalty_s",
    "era_code",
]

# Variance features zeroed for the perfect-strategy counterfactual.
# grid_position and era_code are NOT zeroed — car quality and era are fixed.
VARIANCE_FEATURES = [
    "avg_window_error_laps",
    "window_penalty_s",
    "exec_penalty_s",
]

# ── Real-world F1 champions (ground truth) ────────────────────────────────────
# Used in run_sweep() to populate real_winner — NOT derived from the dataset,
# because actual_points in the RF dataset are race-level and may not sum to the
# correct championship total due to data coverage gaps and driver name variants.
REAL_CHAMPIONS = {
    1994: "Michael Schumacher",
    1995: "Michael Schumacher",
    1996: "Damon Hill",
    1997: "Jacques Villeneuve",
    1998: "Mika Häkkinen",
    1999: "Mika Häkkinen",
    2000: "Michael Schumacher",
    2001: "Michael Schumacher",
    2002: "Michael Schumacher",
    2003: "Michael Schumacher",
    2004: "Michael Schumacher",
    2005: "Fernando Alonso",
    2006: "Fernando Alonso",
    2007: "Kimi Räikkönen",
    2008: "Lewis Hamilton",
    2009: "Jenson Button",
    2010: "Sebastian Vettel",
    2011: "Sebastian Vettel",
    2012: "Sebastian Vettel",
    2013: "Sebastian Vettel",
    2014: "Lewis Hamilton",
    2015: "Lewis Hamilton",
    2016: "Nico Rosberg",
    2017: "Lewis Hamilton",
    2018: "Lewis Hamilton",
    2019: "Lewis Hamilton",
    2020: "Lewis Hamilton",
    2021: "Max Verstappen",
    2022: "Max Verstappen",
    2023: "Max Verstappen",
    2024: "Max Verstappen",
}

COLOUR = {
    "red":    "#e8003d",
    "amber":  "#ffb800",
    "green":  "#00c96e",
    "blue":   "#3d9eff",
    "muted":  "#666666",
    "border": "#2e2e2e",
    "card":   "#1a1a1a",
    "bg":     "#0d0d0d",
    "text":   "#f0f0f0",
}


# ── Model training ─────────────────────────────────────────────────────────────

def train_model(dataset):
    """
    Train RF identically to rf.py — 1994-2011 train, 2012-2024 test.
    Called when f1.py runs standalone (model not passed in from rf.py).
    """
    train = dataset[dataset["year"] <= 2011].copy()
    test  = dataset[dataset["year"] >  2011].copy()

    if train.empty:
        raise ValueError("No training data found (years <= 2011)")

    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train["points_scored"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_r2 = model.score(X_train, y_train)
    cv       = cross_val_score(model, X_train, y_train, cv=5)
    print(f"  Train R²:      {train_r2:.3f}")
    print(f"  Cross-val R²:  {cv.mean():.3f} ± {cv.std():.3f}")

    if not test.empty:
        test_r2 = model.score(
            test[FEATURE_COLS].fillna(0), test["points_scored"]
        )
        print(f"  Test  R²:      {test_r2:.3f}")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print(f"\n  Feature importances:")
    for feat, imp in importances.sort_values(ascending=False).items():
        bar = "█" * int(imp * 40)
        print(f"    {feat:<30} {imp:.3f}  {bar}")

    return model


# ── Counterfactual prediction ──────────────────────────────────────────────────

def predict_perfect_strategy(dataset, model):
    """
    For every driver-race predict points under two scenarios:

      actual:  features as observed (real historical variance)
      perfect: variance features zeroed, grid_position + era_code unchanged

    The difference in predicted points is entirely determined by what the
    RF learned — no correction coefficients are applied manually.

    Returns dataset with added columns:
      rf_predicted_points  — RF prediction using actual features
      rf_perfect_points    — RF prediction with variance features zeroed
      rf_points_gain       — difference (perfect − actual)
    """
    X_actual  = dataset[FEATURE_COLS].fillna(0)
    X_perfect = X_actual.copy()
    for col in VARIANCE_FEATURES:
        X_perfect[col] = 0.0

    dataset = dataset.copy()
    dataset["rf_predicted_points"] = model.predict(X_actual)
    dataset["rf_perfect_points"]   = model.predict(X_perfect)
    dataset["rf_points_gain"]      = (
        dataset["rf_perfect_points"] - dataset["rf_predicted_points"]
    ).round(3)

    return dataset


# ── Championship simulation ────────────────────────────────────────────────────

def simulate_season(year_df):
    """
    Given one season's driver-race rows (with rf predictions already added),
    produce championship standings under both scenarios.

    Uses RF-predicted points so both sides come from the same model —
    the comparison is internally consistent.
    """
    hist = (
        year_df.groupby("driver")["rf_predicted_points"]
        .sum().reset_index()
        .rename(columns={"rf_predicted_points": "predicted_points"})
        .sort_values("predicted_points", ascending=False)
        .reset_index(drop=True)
    )
    hist["predicted_rank"] = hist.index + 1

    perf = (
        year_df.groupby("driver")["rf_perfect_points"]
        .sum().reset_index()
        .rename(columns={"rf_perfect_points": "perfect_points"})
        .sort_values("perfect_points", ascending=False)
        .reset_index(drop=True)
    )
    perf["perfect_rank"] = perf.index + 1

    standings = hist.merge(perf, on="driver")
    standings["points_gain"] = (
        standings["perfect_points"] - standings["predicted_points"]
    ).round(1)
    standings["rank_change"] = (
        standings["predicted_rank"] - standings["perfect_rank"]
    )

    actual = (
        year_df.groupby("driver")["points_scored"]
        .sum().reset_index()
        .rename(columns={"points_scored": "actual_points"})
    )
    standings = standings.merge(actual, on="driver")

    return standings


# ── Multi-year sweep ───────────────────────────────────────────────────────────

def run_sweep(dataset, years):
    sweep = []

    for year in years:
        year_df = dataset[dataset["year"] == year]
        if len(year_df) < 10:
            print(f"  ⚠  {year}: insufficient data ({len(year_df)} rows) — skipping")
            continue

        standings = simulate_season(year_df)

        pred_winner = standings.sort_values("predicted_points", ascending=False).iloc[0]
        perf_winner = standings.sort_values("perfect_points",   ascending=False).iloc[0]

        # Ground-truth champion from lookup — not derived from actual_points
        # (race-level sums are unreliable due to data coverage gaps).
        real_champ  = REAL_CHAMPIONS.get(year, "Unknown")

        pred_flips  = pred_winner["driver"] != real_champ
        perf_flips  = perf_winner["driver"] != real_champ
        title_flips = pred_winner["driver"] != perf_winner["driver"]

        era_label    = year_df["era_label"].iloc[0]
        avg_variance = year_df[VARIANCE_FEATURES].sum(axis=1).mean()
        avg_gain     = year_df["rf_points_gain"].mean()

        print(f"\n  {year} -- {era_label}")
        print(f"    Real champion:              {real_champ}")
        print(f"    RF-predicted champion:      {pred_winner['driver']}  "
              f"{'* PREDICTION CHANGES' if pred_flips else ''}")
        print(f"    Perfect-strategy champion:  {perf_winner['driver']}  "
              f"{'* CHAMPION CHANGES'   if perf_flips else ''}")
        print(f"    Avg variance (s)/race:      {avg_variance:.2f}")
        print(f"    Avg RF points gain:         {avg_gain:.2f}")

        sweep.append({
            "year":            year,
            "era_label":       era_label,
            "real_winner":     real_champ,
            "pred_winner":     pred_winner["driver"],
            "perf_winner":     perf_winner["driver"],
            "pred_flips":      pred_flips,
            "perf_flips":      perf_flips,
            "title_flips":     title_flips,
            "avg_variance_s":  round(avg_variance, 2),
            "avg_points_gain": round(avg_gain, 2),
        })

    return sweep


# ── Plotting ───────────────────────────────────────────────────────────────────

def setup_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family":      "monospace",
        "axes.facecolor":   COLOUR["card"],
        "figure.facecolor": COLOUR["bg"],
        "axes.edgecolor":   COLOUR["border"],
        "text.color":       COLOUR["text"],
        "axes.titlesize":   11,
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "grid.color":       COLOUR["border"],
        "grid.linewidth":   0.5,
    })


def plot_single_season(year, standings, year_df, outputs_dir=None):
    if outputs_dir is None:
        outputs_dir = os.path.join(_HERE, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    setup_style()
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(COLOUR["bg"])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    era_label = year_df["era_label"].iloc[0]
    fig.text(0.02, 0.97, f"F1 {year} — CHAMPIONSHIP SIMULATION",
             fontsize=16, fontweight="bold", color=COLOUR["text"],
             fontfamily="monospace", va="top")
    fig.text(0.02, 0.935,
             f"Era: {era_label}  ·  "
             f"Method: RF counterfactual — variance features zeroed  ·  "
             f"No hardcoded correction factors",
             fontsize=8.5, color=COLOUR["muted"],
             fontfamily="monospace", va="top")

    top10   = standings.head(10)
    drivers = [d[:14] for d in top10["driver"]]
    x       = np.arange(len(top10))
    w       = 0.28

    # ── 1. Three-way points comparison ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - w, top10["actual_points"],    width=w, color=COLOUR["muted"],
            alpha=0.7, label="Actual historical")
    ax1.bar(x,     top10["predicted_points"], width=w, color=COLOUR["blue"],
            alpha=0.85, label="RF predicted")
    ax1.bar(x + w, top10["perfect_points"],   width=w, color=COLOUR["green"],
            alpha=0.85, label="Perfect strategy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(drivers, rotation=30, ha="right", fontsize=7)
    ax1.set_ylabel("Championship points")
    ax1.set_title("Actual vs RF vs Perfect Strategy (Top 10)")
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", alpha=0.3)

    # ── 2. Points gain ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    gain_colors = [COLOUR["green"] if g >= 0 else COLOUR["red"]
                   for g in top10["points_gain"]]
    ax2.bar(x, top10["points_gain"], color=gain_colors, alpha=0.85)
    ax2.axhline(0, color="white", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(drivers, rotation=30, ha="right", fontsize=7)
    ax2.set_ylabel("RF points gained (perfect − predicted)")
    ax2.set_title("Points Gained from Perfect Strategy")
    ax2.grid(axis="y", alpha=0.3)

    # ── 3. Per-driver variance breakdown ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    drv_var = year_df.groupby("driver")[VARIANCE_FEATURES].mean()
    drv_var["total"] = drv_var.sum(axis=1)
    drv_var = drv_var.sort_values("total", ascending=False).head(10)
    y3 = np.arange(len(drv_var))
    ax3.barh(y3, drv_var["window_penalty_s"], color=COLOUR["amber"],
             alpha=0.85, label="Window penalty")
    ax3.barh(y3, drv_var["exec_penalty_s"],
             left=drv_var["window_penalty_s"],
             color=COLOUR["red"], alpha=0.85, label="Exec penalty")
    ax3.set_yticks(y3)
    ax3.set_yticklabels([n[:14] for n in drv_var.index], fontsize=7)
    ax3.set_xlabel("Avg seconds per race")
    ax3.set_title("Avg Strategy Variance by Driver")
    ax3.legend(fontsize=7)
    ax3.grid(axis="x", alpha=0.3)

    # ── 4. Championship rank change ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    rank_colors = [COLOUR["green"] if r > 0 else COLOUR["red"] if r < 0
                   else COLOUR["muted"] for r in top10["rank_change"]]
    ax4.bar(x, top10["rank_change"], color=rank_colors, alpha=0.85)
    ax4.axhline(0, color="white", linewidth=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(drivers, rotation=30, ha="right", fontsize=7)
    ax4.set_ylabel("Championship positions gained")
    ax4.set_title("Championship Rank Change (Perfect Strategy)")
    ax4.grid(axis="y", alpha=0.3)

    # ── 5. RF prediction quality scatter ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(year_df["points_scored"], year_df["rf_predicted_points"],
                color=COLOUR["blue"], alpha=0.5, s=20, label="Actual vs RF")
    ax5.scatter(year_df["points_scored"], year_df["rf_perfect_points"],
                color=COLOUR["green"], alpha=0.5, s=20,
                label="Actual vs Perfect")
    lim = max(year_df["points_scored"].max(),
              year_df["rf_perfect_points"].max()) + 2
    ax5.plot([0, lim], [0, lim], color=COLOUR["muted"],
             linestyle="--", linewidth=0.8)
    ax5.set_xlabel("Actual points scored")
    ax5.set_ylabel("RF predicted points")
    ax5.set_title("RF Prediction Quality (this season)")
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.3)

    # ── 6. Season summary ─────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    pred_winner = standings.sort_values("predicted_points", ascending=False).iloc[0]
    perf_winner = standings.sort_values("perfect_points",   ascending=False).iloc[0]
    title_flips = pred_winner["driver"] != perf_winner["driver"]

    lines = [
        f"Season:              {year}",
        f"Era:                 {era_label}",
        "",
        f"RF-predicted winner:",
        f"  {pred_winner['driver']}",
        f"  {pred_winner['predicted_points']:.1f} pts",
        "",
        f"Perfect-strategy winner:",
        f"  {perf_winner['driver']}",
        f"  {perf_winner['perfect_points']:.1f} pts",
        "",
        f"Title changes: {'★ YES' if title_flips else 'No'}",
        "",
        f"Avg variance/race:",
        f"  {year_df[VARIANCE_FEATURES].sum(axis=1).mean():.2f}s",
        f"Avg RF points gain:",
        f"  {year_df['rf_points_gain'].mean():.2f}",
    ]
    for i, line in enumerate(lines):
        color = COLOUR["amber"] if "★" in line else COLOUR["text"]
        ax6.text(0.05, 0.97 - i * 0.062, line,
                 transform=ax6.transAxes, fontsize=8.5,
                 fontfamily="monospace", color=color, va="top")
    ax6.set_title("Season Summary")

    outfile = os.path.join(outputs_dir, f"f1_simulation_{year}.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=COLOUR["bg"])
    print(f"  ✓ Saved {outfile}")
    plt.show()
    plt.close()


def plot_sweep(sweep_results, dataset, outputs_dir=None):
    if outputs_dir is None:
        outputs_dir = os.path.join(_HERE, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    setup_style()
    df    = pd.DataFrame(sweep_results)
    years = df["year"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(COLOUR["bg"])
    fig.suptitle(
        "F1 CHAMPIONSHIP SIMULATION — PERFECT STRATEGY COUNTERFACTUAL",
        fontsize=13, fontweight="bold", color=COLOUR["text"],
        fontfamily="monospace"
    )

    # ── 1. Avg variance over time ─────────────────────────────────────────────
    ax = axes[0, 0]
    ax.bar(years, df["avg_variance_s"], color=COLOUR["blue"],
           alpha=0.8, label="Avg strategy variance (s)")
    ax.set_title("Average Strategy Variance per Driver-Race")
    ax.set_xlabel("Year"); ax.set_ylabel("Seconds")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # ── 2. Avg RF points gain over time ──────────────────────────────────────
    ax = axes[0, 1]
    ax.fill_between(years, df["avg_points_gain"], alpha=0.25,
                    color=COLOUR["green"])
    ax.plot(years, df["avg_points_gain"], color=COLOUR["green"],
            linewidth=2.5, marker="o", markersize=5)
    ax.set_title("Avg RF Points Gained from Perfect Strategy")
    ax.set_xlabel("Year"); ax.set_ylabel("Points per driver-race")
    ax.grid(axis="y", alpha=0.3)

    # ── 3. Title flip seasons ─────────────────────────────────────────────────
    ax = axes[1, 0]
    flip_years    = [r["year"] for r in sweep_results if r["title_flips"]]
    no_flip_years = [r["year"] for r in sweep_results if not r["title_flips"]]
    ax.scatter(no_flip_years, [0] * len(no_flip_years),
               color=COLOUR["muted"], s=60, zorder=3, label="Same champion")
    ax.scatter(flip_years, [0] * len(flip_years),
               color=COLOUR["amber"], s=140, marker="*", zorder=4,
               label="Title would change")
    for r in sweep_results:
        if r["title_flips"]:
            ax.annotate(r["perf_winner"][:10], (r["year"], 0),
                        textcoords="offset points", xytext=(0, 14),
                        fontsize=7, color=COLOUR["amber"],
                        ha="center", fontfamily="monospace")
    ax.set_title("Seasons Where Perfect Strategy Changes the Champion")
    ax.set_xlabel("Year"); ax.set_yticks([])
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)

    # ── 4. Variance distribution by era (box plot) ───────────────────────────
    ax = axes[1, 1]
    dataset["total_variance"] = dataset[VARIANCE_FEATURES].sum(axis=1)
    era_groups = dataset.groupby("era_label")["total_variance"]
    era_labels_sorted = (
        dataset.drop_duplicates("era_label")
        .sort_values("year")[["year", "era_label"]]
        .set_index("era_label")["year"]
        .sort_values().index.tolist()
    )
    data_by_era = [
        era_groups.get_group(e).values
        for e in era_labels_sorted
        if e in era_groups.groups
    ]
    bp = ax.boxplot(data_by_era, patch_artist=True, notch=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOUR["blue"])
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color(COLOUR["amber"])
        median.set_linewidth(2)
    ax.set_xticklabels(
        [e[:12] for e in era_labels_sorted],
        rotation=20, ha="right", fontsize=7
    )
    ax.set_ylabel("Total variance (s) per driver-race")
    ax.set_title("Strategy Variance Distribution by Era")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(outputs_dir, "f1_sweep_simulation.png")
    plt.savefig(out_path, dpi=150,
                bbox_inches="tight", facecolor=COLOUR["bg"])
    print(f"  ✓ Saved {out_path}")
    plt.show()
    plt.close()


# ── Public API — called from analyser.py after rf.py trains the model ──────────

def run(dataset, model, no_plot=False, outputs_dir=None):
    """
    Entry point when called from the analyser.py pipeline.
    dataset — the full rf_dataset DataFrame already in memory
    model   — the trained RandomForestRegressor from rf.py
    """
    if outputs_dir is None:
        outputs_dir = os.path.join(_HERE, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    print("\n── Championship Simulation ──────────────────────────────────")
    dataset = predict_perfect_strategy(dataset, model)
    years   = sorted(dataset["year"].unique())
    sweep   = run_sweep(dataset, years)

    if sweep:
        df = pd.DataFrame(sweep)
        out_path = os.path.join(outputs_dir, "f1_simulation_sweep.csv")
        df.to_csv(out_path, index=False)
        print(f"\n  ✓ Saved {out_path}")

        pred_flips = [r for r in sweep if r["pred_flips"]]
        perf_flips = [r for r in sweep if r["perf_flips"]]
        print(f"\n  PREDICTION CHANGES (RF != real):    {len(pred_flips)}/{len(sweep)}")
        for r in pred_flips:
            print(f"    {r['year']}  real: {r['real_winner']:<22} RF: {r['pred_winner']}")
        print(f"\n  CHAMPION CHANGES (perfect != real): {len(perf_flips)}/{len(sweep)}")
        for r in perf_flips:
            print(f"    {r['year']}  real: {r['real_winner']:<22} perfect: {r['perf_winner']}")

        if not no_plot:
            plot_sweep(sweep, dataset, outputs_dir)

    return sweep


# ── Standalone entry point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="F1 Championship Simulator v4 — LUT University Thesis"
    )
    parser.add_argument("--year",       type=int,  default=None)
    parser.add_argument("--start",      type=int,  default=1994)
    parser.add_argument("--end",        type=int,  default=2024)
    parser.add_argument("--dataset",    type=str,  default=RF_DATASET_CSV)
    parser.add_argument("--outputs",    type=str,  default=OUTPUTS_DIR,
                        help="Directory for output files (default: ./outputs)")
    parser.add_argument("--no-plot",    action="store_true")
    args = parser.parse_args()

    outputs_dir = args.outputs
    os.makedirs(outputs_dir, exist_ok=True)

    print("\n  F1 Championship Simulator  v4.0")
    print("  LUT University — Bachelor's Thesis")
    print("  Method: RF counterfactual — no hardcoded correction factors\n")

    if not os.path.exists(args.dataset):
        print(f"  ✗ Dataset not found: {args.dataset}")
        print("    Run analyser.py first to generate f1_rf_dataset.csv")
        return

    dataset = pd.read_csv(args.dataset)
    print(f"  ✓ Loaded dataset: {len(dataset)} driver-race rows")
    print(f"    Years:   {dataset['year'].min()}–{dataset['year'].max()}")
    print(f"    Drivers: {dataset['driver'].nunique()}")

    print("\n── Training RF model ────────────────────────────────────────")
    model = train_model(dataset)

    print("\n── Generating counterfactual predictions ────────────────────")
    dataset = predict_perfect_strategy(dataset, model)

    if args.year is not None:
        year_df = dataset[dataset["year"] == args.year]
        if year_df.empty:
            print(f"  ✗ No data for {args.year}")
            return

        print(f"\n── Simulating {args.year} ───────────────────────────────────")
        standings = simulate_season(year_df)

        pred_winner = standings.sort_values("predicted_points", ascending=False).iloc[0]
        perf_winner = standings.sort_values("perfect_points",   ascending=False).iloc[0]
        title_flips = pred_winner["driver"] != perf_winner["driver"]

        print(f"\n  Era: {year_df['era_label'].iloc[0]}")
        print(f"\n  {'Driver':<20} {'Actual':>7} {'RF pred':>8} "
              f"{'Perfect':>8} {'Gain':>6} {'Rank Δ':>7}")
        print(f"  {'-'*60}")
        for _, row in standings.head(10).iterrows():
            print(f"  {row['driver']:<20} "
                  f"{row['actual_points']:>7.0f} "
                  f"{row['predicted_points']:>8.1f} "
                  f"{row['perfect_points']:>8.1f} "
                  f"{row['points_gain']:>+6.1f} "
                  f"{row['rank_change']:>+7.0f}")

        if title_flips:
            print(f"\n  ★ CHAMPIONSHIP OUTCOME CHANGES!")
            print(f"    RF-predicted champion:     {pred_winner['driver']} "
                  f"({pred_winner['predicted_points']:.1f} pts)")
            print(f"    Perfect-strategy champion: {perf_winner['driver']} "
                  f"({perf_winner['perfect_points']:.1f} pts)")
        else:
            print(f"\n  ✓ Same champion: {pred_winner['driver']}")

        if not args.no_plot:
            plot_single_season(args.year, standings, year_df, outputs_dir)

    else:
        years = sorted(dataset[
            dataset["year"].between(args.start, args.end)
        ]["year"].unique())

        print(f"\n── Sweeping {args.start}–{args.end} "
              f"({len(years)} seasons) ───────────────────")

        sweep_results = run_sweep(dataset, years)

        if sweep_results:
            df = pd.DataFrame(sweep_results)
            out_path = os.path.join(outputs_dir, "f1_simulation_sweep.csv")
            df.to_csv(out_path, index=False)
            print(f"\n  ✓ Saved {out_path}")

            pred_flips = [r for r in sweep_results if r["pred_flips"]]
            perf_flips = [r for r in sweep_results if r["perf_flips"]]
            print(f"\n  PREDICTION CHANGES (RF != real):    {len(pred_flips)}/{len(sweep_results)}")
            for r in pred_flips:
                print(f"    {r['year']}  real: {r['real_winner']:<22} RF: {r['pred_winner']}")
            print(f"\n  CHAMPION CHANGES (perfect != real): {len(perf_flips)}/{len(sweep_results)}")
            for r in perf_flips:
                print(f"    {r['year']}  real: {r['real_winner']:<22} perfect: {r['perf_winner']}")

            if not args.no_plot:
                plot_sweep(sweep_results, dataset, outputs_dir)


if __name__ == "__main__":
    main()