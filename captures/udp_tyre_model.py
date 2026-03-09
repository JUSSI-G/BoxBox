"""
UDP Tyre Degradation Model & Pit Window Optimiser
==================================================
Bachelor's Thesis Tool — LUT University
"The impact of software engineering on strategy development in Formula One"

Uses F1 25 UDP telemetry to demonstrate what private telemetry enables
that no public dataset can provide: per-compound degradation curves,
fuel-corrected lap time models, and optimal pit window calculation.

Thesis argument made concrete:
  - "Rich team"  = telemetry model  -> knows optimal pit window
  - "Poor team"  = lap-count heuristic -> suboptimal timing
  - Delta between them = competitive value of software

Usage:
    python udp_tyre_model.py
    python udp_tyre_model.py --csv udp_dump_20260309_134516.csv --race-laps 57
    python udp_tyre_model.py --pit-loss 25.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import argparse
import os

DEFAULT_CSV = "udp_dump_20260309_134516.csv"
RACE_LAPS   = 57
FUEL_START  = 108.0
FUEL_BURN   = 1.618

TEMP_WINDOW = {
    "Soft":   (80, 100),
    "Medium": (85, 105),
    "Hard":   (90, 115),
}

COLOUR = {
    "red":    "#e8003d",
    "amber":  "#ffb800",
    "green":  "#00c96e",
    "blue":   "#3d9eff",
    "soft":   "#e8003d",
    "medium": "#ffb800",
    "hard":   "#cccccc",
    "muted":  "#555555",
    "border": "#2e2e2e",
    "card":   "#1a1a1a",
    "bg":     "#0d0d0d",
    "text":   "#f0f0f0",
}
COMPOUND_COLOUR = {
    "Soft":   COLOUR["soft"],
    "Medium": COLOUR["amber"],
    "Hard":   COLOUR["hard"],
}


def setup_style():
    plt.rcParams.update({
        "font.family":      "monospace",
        "axes.facecolor":   COLOUR["card"],
        "figure.facecolor": COLOUR["bg"],
        "axes.edgecolor":   COLOUR["border"],
        "axes.labelcolor":  COLOUR["muted"],
        "xtick.color":      COLOUR["muted"],
        "ytick.color":      COLOUR["muted"],
        "grid.color":       COLOUR["border"],
        "grid.linewidth":   0.5,
        "text.color":       COLOUR["text"],
        "axes.titlecolor":  COLOUR["text"],
        "axes.titlesize":   10,
        "axes.labelsize":   9,
    })


def load_and_clean(path):
    df = pd.read_csv(path)
    wear_cols  = ["tyre_wear_fl", "tyre_wear_fr", "tyre_wear_rl", "tyre_wear_rr"]
    surf_cols  = ["tyre_surf_fl", "tyre_surf_fr", "tyre_surf_rl", "tyre_surf_rr"]
    inner_cols = ["tyre_inner_fl","tyre_inner_fr","tyre_inner_rl","tyre_inner_rr"]
    df["avg_wear"]  = df[wear_cols].mean(axis=1)
    df["avg_surf"]  = df[surf_cols].mean(axis=1)
    df["avg_inner"] = df[inner_cols].mean(axis=1)

    def is_realistic(grp):
        vals = grp[wear_cols].values.flatten()
        return len([v for v in vals if 0.0 < v < 99.0]) >= 5

    good_drivers = [d for d, g in df.groupby("driver_name") if is_realistic(g)]
    clean = df[
        df["driver_name"].isin(good_drivers) &
        (df["tyres_age_laps"] > 0) &
        (df["lap_time_s"].between(85, 115)) &
        (df["lap_invalid"] == 0)
    ].copy()
    return df, clean, good_drivers


def fit_compound_models(clean_df):
    models = {}
    print("\n-- Degradation Models (fitted from UDP telemetry) --------------")
    print(f"  {'Compound':<10} {'Base(s)':>8} {'Deg/lap':>9} {'Fuel/kg':>9} {'R2':>6} {'Wear%/lap':>10} {'n':>5}")
    print(f"  {'-'*60}")

    for compound in ["Hard", "Medium", "Soft"]:
        cdf = clean_df[clean_df["actual_compound"] == compound].copy()
        if len(cdf) < 5:
            continue
        X = np.column_stack([np.ones(len(cdf)), cdf["tyres_age_laps"].values, cdf["fuel_in_tank"].values])
        y = cdf["lap_time_s"].values
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        base, deg, fuel = coeffs
        y_pred = X @ coeffs
        r2 = max(0.0, 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2))

        wear_rates = []
        for (drv, stint), grp in cdf.groupby(["driver_name", "stint"]):
            grp = grp.sort_values("lap")
            dw = grp["avg_wear"].diff().dropna()
            dw = dw[dw.between(0.05, 5.0)]
            if len(dw):
                wear_rates.append(dw.mean())
        wear_per_lap = float(np.mean(wear_rates)) if wear_rates else 0.6

        models[compound] = {
            "base": base, "deg_per_lap": deg, "fuel_effect": fuel,
            "r2": r2, "n": len(cdf), "wear_per_lap": wear_per_lap,
        }
        print(f"  {compound:<10} {base:>8.3f} {deg:>+9.4f} {fuel:>+9.4f} {r2:>6.3f} {wear_per_lap:>10.3f} {len(cdf):>5}")
    return models


def simulate_strategy(strategy, models, race_laps=RACE_LAPS,
                      fuel_start=FUEL_START, fuel_burn=FUEL_BURN, pit_loss_s=22.0):
    """
    Simulate race lap times for a given strategy.
    Pit loss is added to total race time but NOT injected into lap_times,
    so lap_times contains only clean racing laps with no spikes.
    """
    driven = sum(n for _, n in strategy)
    if driven != race_laps:
        last_c, _ = strategy[-1]
        strategy = strategy[:-1] + [(last_c, race_laps - sum(n for _, n in strategy[:-1]))]

    lap_times, pit_laps, fuel, n_stops = [], [], fuel_start, 0

    for stint_idx, (compound, num_laps) in enumerate(strategy):
        m = models.get(compound)
        if stint_idx > 0:
            pit_laps.append(sum(n for _, n in strategy[:stint_idx]))
            n_stops += 1
        for age in range(num_laps):
            t = (m["base"] + m["deg_per_lap"] * age + m["fuel_effect"] * fuel) if m else 91.0
            lap_times.append(t)
            fuel -= fuel_burn

    total_time = sum(lap_times) + n_stops * pit_loss_s
    return total_time, lap_times, pit_laps


def find_optimal_strategy(models, race_laps=RACE_LAPS, top_n=5):
    compounds = list(models.keys())
    results   = []

    # 1-stop: must use 2 different compounds
    for c1 in compounds:
        for c2 in compounds:
            if c1 == c2:
                continue
            for split in range(10, race_laps - 10):
                strat = [(c1, split), (c2, race_laps - split)]
                t, laps, pits = simulate_strategy(strat, models, race_laps)
                results.append((t, strat, pits))

    # 2-stop: no consecutive same compound, at least 2 different total
    for c1 in compounds:
        for c2 in compounds:
            for c3 in compounds:
                if c1 == c2 or c2 == c3:
                    continue
                if len({c1, c2, c3}) < 2:
                    continue
                for s1 in range(8, race_laps - 16):
                    for s2 in range(8, race_laps - s1 - 8):
                        s3 = race_laps - s1 - s2
                        if s3 < 8:
                            continue
                        strat = [(c1, s1), (c2, s2), (c3, s3)]
                        t, laps, pits = simulate_strategy(strat, models, race_laps)
                        results.append((t, strat, pits))

    results.sort(key=lambda x: x[0])
    return results[:top_n]


def heuristic_strategy(models, race_laps=RACE_LAPS):
    """Poor team: split race in half using two most durable compounds."""
    ranked = sorted(models.keys(), key=lambda c: models[c]["deg_per_lap"])
    c1 = ranked[0]
    c2 = ranked[1] if len(ranked) > 1 else ranked[0]
    split = race_laps // 2
    strat = [(c1, split), (c2, race_laps - split)]
    t, laps, pits = simulate_strategy(strat, models, race_laps)
    return t, strat, pits, laps


def analyse_temp_windows(df, good_drivers):
    results = []
    for drv in good_drivers:
        grp = df[df["driver_name"] == drv].copy()
        for compound in ["Hard", "Medium", "Soft"]:
            cgrp = grp[grp["actual_compound"] == compound]
            if len(cgrp) < 3:
                continue
            lo, hi  = TEMP_WINDOW[compound]
            in_w    = cgrp[(cgrp["avg_surf"] >= lo) & (cgrp["avg_surf"] <= hi)]
            out_w   = cgrp[(cgrp["avg_surf"] < lo)  | (cgrp["avg_surf"] > hi)]
            results.append({
                "driver": drv, "compound": compound,
                "laps_in": len(in_w), "laps_out": len(out_w),
                "pct_in": 100 * len(in_w) / len(cgrp),
                "avg_temp": cgrp["avg_surf"].mean(),
            })
    return pd.DataFrame(results)


def mask_pit_laps(lap_times, pit_laps):
    """Replace pit lap with None so matplotlib shows a gap instead of a spike."""
    masked = [float(t) for t in lap_times]
    for p in pit_laps:
        idx = p - 1
        if 0 <= idx < len(masked):
            masked[idx] = None
    return masked


def cumulative_with_pits(lap_times, pit_laps, pit_loss, total_laps):
    """Cumulative race time including pit losses inserted at the correct lap."""
    cum, running = [], 0.0
    for i, t in enumerate(lap_times[:total_laps]):
        running += t
        if (i + 1) in pit_laps:
            running += pit_loss
        cum.append(running)
    return cum


def plot_all(df, clean_df, models, good_drivers, optimal, heuristic_result,
             race_laps=RACE_LAPS, pit_loss_s=22.0):

    setup_style()
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(COLOUR["bg"])
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

    fig.text(0.02, 0.975, "F1 TYRE DEGRADATION MODEL -- UDP TELEMETRY ANALYSIS",
             fontsize=16, fontweight="bold", color=COLOUR["text"],
             fontfamily="monospace", va="top")
    fig.text(0.02, 0.945,
             f"Source: F1 25 UDP  |  {race_laps} laps  |  "
             f"Track: {df['track_length_m'].iloc[0]/1000:.1f}km  |  "
             f"Valid telemetry drivers: {', '.join(good_drivers)}",
             fontsize=8, color=COLOUR["muted"], fontfamily="monospace", va="top")

    laps_x = list(range(1, race_laps + 1))

    # 1. Lap time vs tyre age
    ax = fig.add_subplot(gs[0, :2])
    for compound, col in COMPOUND_COLOUR.items():
        cdf = clean_df[clean_df["actual_compound"] == compound]
        if cdf.empty: continue
        ax.scatter(cdf["tyres_age_laps"], cdf["lap_time_s"],
                   color=col, alpha=0.45, s=18, label=f"{compound} data")
        m = models.get(compound)
        if m:
            ages = np.linspace(0, cdf["tyres_age_laps"].max(), 60)
            pred = m["base"] + m["deg_per_lap"] * ages + m["fuel_effect"] * clean_df["fuel_in_tank"].mean()
            ax.plot(ages, pred, color=col, linewidth=2.5, linestyle="--",
                    label=f"{compound} fit (+{m['deg_per_lap']*1000:.1f}ms/lap)")
    ax.set_xlabel("Tyre age (laps)"); ax.set_ylabel("Lap time (s)")
    ax.set_title("Lap Time vs Tyre Age per Compound (fuel-corrected trend)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # 2. Tyre wear accumulation
    ax = fig.add_subplot(gs[0, 2])
    for drv in good_drivers:
        grp = df[df["driver_name"] == drv].sort_values("lap")
        for compound, col in COMPOUND_COLOUR.items():
            cgrp = grp[grp["actual_compound"] == compound]
            if len(cgrp) < 2: continue
            ax.scatter(cgrp["tyres_age_laps"], cgrp["avg_wear"], color=col, s=15, alpha=0.7)
    handles = [Line2D([0],[0], marker="o", color="w",
                      markerfacecolor=COMPOUND_COLOUR[c], label=c, markersize=8)
               for c in COMPOUND_COLOUR if c in models]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel("Tyre age (laps)"); ax.set_ylabel("Avg wear (%)")
    ax.set_title("Tyre Wear Accumulation"); ax.grid(alpha=0.3)

    # 3. Fuel effect
    ax = fig.add_subplot(gs[0, 3])
    for compound, col in COMPOUND_COLOUR.items():
        cdf = clean_df[clean_df["actual_compound"] == compound]
        m   = models.get(compound)
        if len(cdf) < 5 or m is None: continue
        corrected = cdf["lap_time_s"] - m["deg_per_lap"] * cdf["tyres_age_laps"]
        ax.scatter(cdf["fuel_in_tank"], corrected, color=col, s=15, alpha=0.6, label=compound)
        fuels = np.linspace(cdf["fuel_in_tank"].min(), cdf["fuel_in_tank"].max(), 40)
        ax.plot(fuels, m["base"] + m["fuel_effect"] * fuels, color=col, linewidth=1.5, linestyle="--")
    ax.set_xlabel("Fuel in tank (kg)"); ax.set_ylabel("Age-corrected lap time (s)")
    ax.set_title("Fuel Effect on Lap Time"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 4. Strategy comparison — clean lines, no spike
    ax = fig.add_subplot(gs[1, :2])
    h_time, h_strat, h_pits, h_lap_times = heuristic_result
    opt_time, opt_strat, opt_pits         = optimal[0]
    _, opt_lap_times, _                   = simulate_strategy(opt_strat, models, race_laps,
                                                               pit_loss_s=pit_loss_s)
    h_masked   = mask_pit_laps(h_lap_times[:race_laps],   h_pits)
    opt_masked = mask_pit_laps(opt_lap_times[:race_laps], opt_pits)

    ax.plot(laps_x, h_masked,   color=COLOUR["red"],   linewidth=1.8,
            label=f"Heuristic: {h_time:.1f}s total")
    ax.plot(laps_x, opt_masked, color=COLOUR["green"], linewidth=1.8,
            label=f"Optimal:   {opt_time:.1f}s total")

    for p in h_pits:
        ax.axvline(p, color=COLOUR["red"],   linestyle=":", alpha=0.6, linewidth=1.2)
        ax.text(p + 0.3, 88.8, f"pit L{p}", fontsize=7, color=COLOUR["red"],   rotation=90, va="bottom")
    for p in opt_pits:
        ax.axvline(p, color=COLOUR["green"], linestyle=":", alpha=0.6, linewidth=1.2)
        ax.text(p + 0.3, 89.5, f"pit L{p}", fontsize=7, color=COLOUR["green"], rotation=90, va="bottom")

    delta = h_time - opt_time
    ax.set_xlabel("Lap"); ax.set_ylabel("Lap time (s)")
    ax.set_title(f"Heuristic vs Telemetry-Optimal Strategy  (delta = {delta:+.1f}s)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 5. Top 5 strategies
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off"); ax.set_facecolor(COLOUR["card"])
    ax.set_title("Top 5 Strategies (telemetry model)", fontsize=9)
    lines = []
    for rank, (t, strat, pits) in enumerate(optimal[:5], 1):
        strat_str = " -> ".join(f"{c[:3]}x{n}" for c, n in strat)
        pit_str   = ", ".join(f"L{p}" for p in pits)
        lines.append(f"#{rank}  {strat_str}")
        lines.append(f"    {t:.1f}s  pits@{pit_str}  ({t - optimal[0][0]:+.1f}s)")
        lines.append("")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=8,
            fontfamily="monospace", color=COLOUR["text"], va="top", linespacing=1.5)

    # 6. Model coefficients
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off"); ax.set_facecolor(COLOUR["card"])
    ax.set_title("Model Coefficients (from UDP telemetry)", fontsize=9)
    lines = ["Compound  Deg/lap   Wear%/lap  R2\n"]
    for compound, m in models.items():
        lines.append(f"{compound:<8}  {m['deg_per_lap']:>+.4f}s  {m['wear_per_lap']:>6.3f}%    {m['r2']:.3f}")
    lines += ["", f"Fuel burn:  {FUEL_BURN:.4f} kg/lap",
              f"Pit loss:   {pit_loss_s:.1f}s", "",
              "These signals are NOT in any",
              "public F1 dataset. Game UDP",
              "telemetry fills this data gap.",
              "",
              "Low R2 reflects real noise:",
              "traffic, track evolution,",
              "driver variation — same gaps",
              "real teams close with 250+",
              "sensors and ML models."]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=8,
            fontfamily="monospace", color=COLOUR["text"], va="top", linespacing=1.5)

    # 7. Temperature window
    ax = fig.add_subplot(gs[2, :2])
    temp_df = analyse_temp_windows(df, good_drivers)
    if not temp_df.empty:
        for i, (_, row) in enumerate(temp_df.iterrows()):
            col = COMPOUND_COLOUR.get(row["compound"], COLOUR["blue"])
            ax.bar(i, row["laps_in"],  color=col, alpha=0.8)
            ax.bar(i, row["laps_out"], bottom=row["laps_in"], color=COLOUR["red"], alpha=0.5)
        ax.set_xticks(range(len(temp_df)))
        ax.set_xticklabels([f"{r['driver'][:8]}\n{r['compound']}" for _, r in temp_df.iterrows()],
                           rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Laps")
        ax.set_title("Laps in Optimal Temp Window (coloured) vs Outside (red)\n"
                     "Real teams monitor this live — not available in public data")
        ax.grid(axis="y", alpha=0.3)

    # 8. Cumulative software advantage
    ax = fig.add_subplot(gs[2, 2:])
    cum_opt  = cumulative_with_pits(opt_lap_times, opt_pits, pit_loss_s, race_laps)
    cum_heur = cumulative_with_pits(h_lap_times,   h_pits,   pit_loss_s, race_laps)
    delta_cum = [h - o for h, o in zip(cum_heur, cum_opt)]

    ax.fill_between(laps_x, 0, delta_cum, alpha=0.2, color=COLOUR["green"])
    ax.plot(laps_x, delta_cum, color=COLOUR["green"], linewidth=2)
    ax.axhline(0, color=COLOUR["border"], linewidth=1)

    for p in opt_pits:
        ax.axvline(p, color=COLOUR["green"], linestyle="--", alpha=0.5, linewidth=1,
                   label="Optimal pit" if p == opt_pits[0] else "")
    for p in h_pits:
        ax.axvline(p, color=COLOUR["red"], linestyle="--", alpha=0.5, linewidth=1,
                   label="Heuristic pit" if p == h_pits[0] else "")

    ax.set_xlabel("Lap"); ax.set_ylabel("Cumulative time advantage (s)")
    ax.set_title(f"Software Advantage Accumulation Over Race Distance\n"
                 f"Telemetry model vs lap-count heuristic  |  "
                 f"Final: {delta:+.1f}s = ~{delta/1.5:.1f} positions")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.savefig("udp_tyre_model.png", dpi=150, bbox_inches="tight", facecolor=COLOUR["bg"])
    print("\n  Saved udp_tyre_model.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="UDP Tyre Degradation Model -- LUT University Bachelor's Thesis"
    )
    parser.add_argument("--csv",       default=DEFAULT_CSV)
    parser.add_argument("--race-laps", type=int,   default=RACE_LAPS)
    parser.add_argument("--pit-loss",  type=float, default=22.0)
    args = parser.parse_args()

    print("\n  UDP Tyre Degradation Model & Pit Window Optimiser")
    print("  LUT University -- Bachelor's Thesis\n")

    if not os.path.exists(args.csv):
        print(f"  ERROR: CSV not found: {args.csv}")
        print(f"  Place {args.csv} in the same folder as this script.")
        return

    print(f"  Loading: {args.csv}")
    df, clean_df, good_drivers = load_and_clean(args.csv)
    print(f"  Laps: {df['lap'].max()}  |  Valid telemetry drivers: {good_drivers}")
    print(f"  Track: {df['track_length_m'].iloc[0]/1000:.1f}km  |  "
          f"Compounds: {sorted(df['actual_compound'].unique())}")

    models = fit_compound_models(clean_df)
    if len(models) < 2:
        print("\n  ERROR: Need at least 2 compounds with valid data.")
        return

    print("\n-- Optimal Strategy Search -------------------------------------")
    optimal = find_optimal_strategy(models, args.race_laps)
    opt_time, opt_strat, opt_pits = optimal[0]
    print(f"  Best: {' -> '.join(f'{c}x{n}' for c,n in opt_strat)}")
    print(f"  Total: {opt_time:.1f}s  |  Pit laps: {opt_pits}")

    print("\n-- Heuristic Strategy (no telemetry model) ---------------------")
    heuristic_result = heuristic_strategy(models, args.race_laps)
    h_time, h_strat, h_pits, _ = heuristic_result
    print(f"  Strategy: {' -> '.join(f'{c}x{n}' for c,n in h_strat)}")
    print(f"  Total: {h_time:.1f}s  |  Pit lap: {h_pits}")

    delta = h_time - opt_time
    print(f"\n  Software advantage: {delta:+.1f}s (~{delta/1.5:.1f} positions)")
    print(f"  Derived from tyre degradation telemetry not in public data.")

    print("\n  Generating charts...")
    plot_all(df, clean_df, models, good_drivers, optimal, heuristic_result,
             args.race_laps, args.pit_loss)

    print("\n-- Penalty constant for testi.py -------------------------------")
    hard    = models.get("Hard", {})
    implied = (hard.get("deg_per_lap", 0.0325) / hard.get("wear_per_lap", 0.604)) * 1.5
    print(f"  UDP-calibrated value: {implied:.4f}s per lap of window error")
    print(f"  Update UDP_DEG_COEFFICIENT in testi.py to this value.")


if __name__ == "__main__":
    main()