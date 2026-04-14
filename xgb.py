"""
F1 Strategy Outcome Predictor — XGBoost
=========================================
Bachelor's Thesis Tool — LUT University
"The impact of software engineering on strategy development in Formula One"

XGBoost model for predicting race points from strategy variance features.
Default model — use rf.py for Random Forest baseline comparison.

Usage: triggered by analyser.py (default) or analyser.py --model xgb
       Use analyser.py --model rf for Random Forest baseline.
"""

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("XGBoost not installed — run: pip install xgboost")
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, sys, json
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

# Ensure sibling files are importable regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

COLOUR = {
    "red":    "#e8003d",
    "amber":  "#ffb800",
    "green":  "#00c96e",
    "blue":   "#3d9eff",
    "muted":  "#666666",
    "card":   "#1a1a1a",
    "bg":     "#0d0d0d",
    "text":   "#f0f0f0",
}

FEATURE_COLS = [
    "grid_position",          # starting position; proxy for car quality (Heilmeier et al. 2018)
    "avg_window_error_laps",  # laps deviated from optimal pit window (Heilmeier et al. 2018)
    "window_penalty_s",       # time cost of pit window error (Heilmeier et al. 2018)
    "exec_penalty_s",         # pit stop execution time vs fastest in race (Heilmeier et al. 2018)
    "era_code",               # ordinal software era encoding (author classification)
    "wet_race",               # 1 if wet surface during race (Phillips 2014; FastF1 2018+)
    "sc_laps",                # safety car laps (FastF1 2018+; 0 for pre-2018)
]

ERA_LABELS_ORDERED = [
    "Pre-digital", "Early electronics", "Telemetry begins",
    "Real-time data", "Simulation tools", "Full analytics",
    "Predictive models", "AI / Monte Carlo"
]

# ── Step 1: Build dataset ──────────────────────────────────────────────────────

def build_dataset(pitstops_df, ergast_results, years, grid_positions=None, race_conditions=None):
    from analyser import (
        compute_pit_window_errors,
        compute_exec_penalties,
        compute_undercut_missed,
        get_era_label,
        WET_RACE_ROUNDS,
    )

    if grid_positions is None:
        grid_positions = {}
    if race_conditions is None:
        race_conditions = {}

    era_encoder = LabelEncoder()
    era_encoder.fit(ERA_LABELS_ORDERED)

    all_rows = []

    for year in years:
        print(f"  Building dataset for {year}...")

        window_errors  = compute_pit_window_errors(pitstops_df, year)
        window_errors  = compute_undercut_missed(window_errors)
        exec_penalties = compute_exec_penalties(pitstops_df, year)

        if window_errors.empty:
            print(f"    ⚠  No pit window data for {year}")
            continue

        year_results = ergast_results[ergast_results["year"] == year]
        if year_results.empty:
            print(f"    ⚠  No Ergast data for {year}")
            continue

        era_label = get_era_label(year)
        era_code  = int(era_encoder.transform([era_label])[0])

        for _, we_row in window_errors.iterrows():
            driver = we_row["driver"]
            race   = we_row["race"]

            ep = exec_penalties[
                (exec_penalties["driver"] == driver) &
                (exec_penalties["race"]   == race)
            ]
            exec_pen = float(ep["exec_penalty_s"].iloc[0]) if not ep.empty else 0.0

            driver_last = driver.split()[-1].lower()
            result = year_results[
                year_results["driver_name"].str.lower().str.contains(driver_last, na=False)
            ]
            if result.empty:
                continue

            points   = float(result["points"].iloc[0])
            driver_last = driver.split()[-1].lower()
            round_num   = int(result["round"].iloc[0]) if "round" in result.columns else 0
            grid_key    = f"{year}_{round_num}_{driver_last}"
            grid_pos    = grid_positions.get(grid_key, 20)

            circuit_key = race.lower().replace(" ", "_")
            year_conds  = race_conditions.get(year, {})
            cond = year_conds.get(circuit_key)
            if cond is None:
                for key, val in year_conds.items():
                    if key in circuit_key or circuit_key in key:
                        cond = val
                        break
            if cond is not None:
                wet_race = int(cond["wet_race"])
                sc_laps  = int(cond["sc_laps"])
            else:
                wet_race = 1 if (year, round_num) in WET_RACE_ROUNDS else 0
                sc_laps  = 0

            window_pen    = float(we_row["window_penalty_s"])
            missed_uc     = float(we_row["missed_undercut_s"])
            total_var     = window_pen + exec_pen + missed_uc

            all_rows.append({
                "year":                  year,
                "era_code":              era_code,
                "era_label":             era_label,
                "race":                  race,
                "driver":                driver,
                "grid_position":         grid_pos,
                "avg_window_error_laps": float(we_row["avg_window_error_laps"]),
                "window_penalty_s":      window_pen,
                "exec_penalty_s":        exec_pen,
                "missed_undercut_s":     missed_uc,
                "total_variance_s":      round(total_var, 3),
                "wet_race":              wet_race,
                "sc_laps":               sc_laps,
                "points_scored":         points,
                "finished_in_points":    int(points > 0),
            })

    df = pd.DataFrame(all_rows)
    print(f"\n  Dataset: {len(df)} driver-race rows across {len(years)} seasons")
    return df


# ── Step 2: Train and evaluate ─────────────────────────────────────────────────

def train_and_evaluate(dataset):
    train = dataset[dataset["year"] <= 2011].copy()
    test  = dataset[dataset["year"] >  2011].copy()

    if train.empty or test.empty:
        print("  ✗ Not enough data to split train/test")
        return None, None

    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train["points_scored"]
    X_test  = test[FEATURE_COLS].fillna(0)
    y_test  = test["points_scored"]

    print(f"\n  Training on {len(train)} rows (1994-2011)")
    print(f"  Testing  on {len(test)} rows (2012-2024)")

    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score  = model.score(X_test,  y_test)
    cv_scores   = cross_val_score(model, X_train, y_train, cv=5)

    print(f"\n  Train R²:          {train_score:.3f}")
    print(f"  Test  R²:          {test_score:.3f}")
    print(f"  Cross-val R²:      {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)

    print(f"\n  Feature importances:")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 40)
        print(f"    {feat:<35} {imp:.3f}  {bar}")

    return model, importances


# ── Step 3: Era importance analysis ───────────────────────────────────────────

def analyse_era_importance(dataset):
    """
    Train separate models per era.
    Compares strategy variance importance vs grid position per era.
    A decreasing variance/grid ratio over time = software reduced strategy errors.

    Returns results dict AND saves to f1_era_importance.json for the dashboard.
    """
    era_groups = {
        "Pre-software (1994-1999)":      (1994, 1999),
        "Early tools (2000-2005)":       (2000, 2005),
        "Full analytics (2006-2011)":    (2006, 2011),
        "Predictive models (2012-2017)": (2012, 2017),
        "AI / Monte Carlo (2018-2024)":  (2018, 2024),
    }

    results = {}

    for era_name, (era_start, era_end) in era_groups.items():
        era_data = dataset[
            (dataset["year"] >= era_start) &
            (dataset["year"] <= era_end)
        ].copy()

        if len(era_data) < 50:
            print(f"  ⚠  Not enough data for {era_name} ({len(era_data)} rows)")
            continue

        X = era_data[FEATURE_COLS].fillna(0)
        y = era_data["points_scored"]

        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)

        importances      = pd.Series(model.feature_importances_, index=FEATURE_COLS)
        variance_imp     = importances[[
            "avg_window_error_laps", "window_penalty_s", "exec_penalty_s"
        ]].sum()
        grid_imp         = importances["grid_position"]
        ratio            = variance_imp / grid_imp if grid_imp > 0 else 0

        # Store all per-feature importances too, for the dashboard
        results[era_name] = {
            "era_start":           era_start,
            "era_end":             era_end,
            "variance_importance": round(float(variance_imp), 4),
            "grid_importance":     round(float(grid_imp), 4),
            "ratio":               round(float(ratio), 4),
            "n_rows":              int(len(era_data)),
            "feature_importances": {
                feat: round(float(imp), 4)
                for feat, imp in importances.items()
            },
        }

        print(f"\n  {era_name} (n={len(era_data)})")
        print(f"    Strategy variance importance: {variance_imp:.3f}")
        print(f"    Grid position importance:     {grid_imp:.3f}")
        print(f"    Variance/Grid ratio:          {ratio:.3f}")
        print(f"    → {'Variance > Car' if variance_imp > grid_imp else 'Car > Variance'}")

    return results


# ── Step 4: Save era importance to JSON ────────────────────────────────────────

def save_era_importance(era_results, outputs_dir):
    """Save era importance dict to JSON so server.py can serve it to the dashboard."""
    out_path = os.path.join(outputs_dir, "f1_era_importance.json")
    with open(out_path, "w") as f:
        json.dump(era_results, f, indent=2)
    print(f"\n  ✓ Saved {out_path}")


# ── Step 5: Plot ───────────────────────────────────────────────────────────────

def plot_results(importances, era_results, outputs_dir=None):
    if outputs_dir is None:
        outputs_dir = os.path.join(_HERE, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family":      "monospace",
        "axes.facecolor":   COLOUR["card"],
        "figure.facecolor": COLOUR["bg"],
        "text.color":       COLOUR["text"],
        "axes.titlesize":   11,
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(COLOUR["bg"])
    fig.suptitle(
        "XGBOOST — STRATEGY VARIANCE IMPACT ON RACE OUTCOMES",
        fontsize=13, fontweight="bold", color=COLOUR["text"],
        fontfamily="monospace"
    )

    # 1. Overall feature importance
    ax = axes[0]
    colors = [
        COLOUR["amber"] if any(k in f for k in ["window","exec","undercut","variance"])
        else COLOUR["blue"]
        for f in importances.index
    ]
    ax.barh(range(len(importances)), importances.values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances.index, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title("Overall Feature Importance\n(amber = strategy variance)")
    ax.grid(axis="x", alpha=0.3)

    # 2. Variance vs grid importance by era
    ax = axes[1]
    if era_results:
        era_names = list(era_results.keys())
        var_imp   = [era_results[e]["variance_importance"] for e in era_names]
        grid_imp  = [era_results[e]["grid_importance"]     for e in era_names]
        x = np.arange(len(era_names))
        w = 0.35
        ax.bar(x - w/2, var_imp,  width=w, color=COLOUR["amber"],
               alpha=0.85, label="Strategy variance")
        ax.bar(x + w/2, grid_imp, width=w, color=COLOUR["blue"],
               alpha=0.85, label="Grid position")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [e.split("(")[0].strip() for e in era_names],
            rotation=15, ha="right", fontsize=8
        )
        ax.set_ylabel("Summed importance")
        ax.set_title("Variance vs Car Quality by Era")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # 3. Variance/grid ratio (key thesis chart)
    ax = axes[2]
    if era_results:
        ratios = [era_results[e]["ratio"] for e in era_names]
        bar_colors = [
            COLOUR["red"]   if r > 1.0 else
            COLOUR["amber"] if r > 0.5 else
            COLOUR["green"]
            for r in ratios
        ]
        ax.bar(x, ratios, color=bar_colors, alpha=0.85)
        ax.axhline(1.0, color="white", linestyle="--", linewidth=1,
                   label="Variance = Car (1.0)")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [e.split("(")[0].strip() for e in era_names],
            rotation=15, ha="right", fontsize=8
        )
        ax.set_ylabel("Variance / Grid importance")
        ax.set_title("Strategy Variance Relative Impact\n(decreasing = software helping)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(outputs_dir, "f1_rf_results.png")
    plt.savefig(out_path, dpi=150,
                bbox_inches="tight", facecolor=COLOUR["bg"])
    print(f"\n  ✓ Saved {out_path}")
    plt.show()
    plt.close()


# ── Entry point ────────────────────────────────────────────────────────────────

def run(pitstops_df, ergast_results_by_year, grid_positions=None, race_conditions=None, start=1994, end=2010, no_plot=False, outputs_dir=None):
    years = list(range(start, end + 1))

    if outputs_dir is None:
        outputs_dir = os.path.join(_HERE, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    all_ergast = pd.concat(
        [df for df in ergast_results_by_year.values() if not df.empty],
        ignore_index=True
    )

    print("\n── Building Dataset ─────────────────────────────────────────")
    dataset = build_dataset(pitstops_df, all_ergast, years, grid_positions or {}, race_conditions or {})
    if dataset.empty:
        print("  ✗ Could not build dataset")
        return

    dataset_path = os.path.join(outputs_dir, "f1_rf_dataset.csv")
    dataset.to_csv(dataset_path, index=False)
    print(f"  ✓ Saved {dataset_path}")

    print("\n── Overall Model ────────────────────────────────────────────")
    model, importances = train_and_evaluate(dataset)
    if model is None:
        return

    print("\n── Era-by-Era Importance Analysis ───────────────────────────")
    era_results = analyse_era_importance(dataset)

    # ── Save era importance JSON for the dashboard ─────────────────────────────
    save_era_importance(era_results, outputs_dir)

    # ── Championship simulation ────────────────────────────────────────────────
    try:
        from f1 import run as run_simulation
        run_simulation(dataset, model, no_plot=no_plot, outputs_dir=outputs_dir)
    except ImportError:
        print("  ⚠  f1.py not found — skipping championship simulation")

    return model, importances, era_results