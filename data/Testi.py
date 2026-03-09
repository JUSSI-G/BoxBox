"""
F1 Championship Strategy Variance Analyser
===========================================
Bachelor's Thesis Tool — LUT University
"The impact of software engineering on strategy development in Formula One"

Research question:
    How much did strategy variance (suboptimal pit windows, missed safety cars,
    poor tyre management) influence championship outcomes — and how does this
    variance decrease as software sophistication increases across eras?

Data sources:
    - Ergast API  : race results, championship standings, pit stop data (1950–2024)
    - pitstops.csv: detailed pit stop times + lap numbers (1994–2010)
    - lap_times.xlsx: theoretical lap pace multipliers (1950–2023)

Usage:
    pip install requests pandas numpy matplotlib openpyxl
    python f1_strategy_analyser.py

    or for a specific season:
    python f1_strategy_analyser.py --year 2010

    or full historical sweep:
    python f1_strategy_analyser.py --mode sweep --start 1994 --end 2010
"""

import requests
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import json
import time
import argparse
import os
import sys
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────

ERGAST_BASE  = "https://api.jolpi.ca/ergast/f1"
CACHE_DIR    = "f1_cache"
PITSTOPS_CSV = "pitstops.csv"      # your file
LAP_TIMES_XL = "Kopio__F1_Lap_Time_Progression.xlsx"

# Visual style
plt.style.use("dark_background")
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

# Points systems by era
POINTS_SYSTEMS = {
    # year_from: {position: points}
    1950: {1:8, 2:6, 3:4, 4:3, 5:2},
    1960: {1:8, 2:6, 3:4, 4:3, 5:2},
    1961: {1:9, 2:6, 3:4, 4:3, 5:2},
    2003: {1:10, 2:8, 3:6, 4:5, 5:4, 6:3, 7:2, 8:1},
    2010: {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1},
}

def get_points_system(year):
    applicable = {y: v for y, v in POINTS_SYSTEMS.items() if y <= year}
    return applicable[max(applicable.keys())]

# Software era model — what fraction of suboptimal decisions get corrected
SOFTWARE_ERAS = {
    # year: {description, window_correction, sc_awareness, variance_reduction}
    1950: {"label": "Pre-digital",       "window_correction": 0.00, "sc_awareness": 0.05, "variance_reduction": 0.00},
    1975: {"label": "Early electronics", "window_correction": 0.08, "sc_awareness": 0.10, "variance_reduction": 0.05},
    1984: {"label": "Telemetry begins",  "window_correction": 0.15, "sc_awareness": 0.20, "variance_reduction": 0.12},
    1994: {"label": "Real-time data",    "window_correction": 0.22, "sc_awareness": 0.35, "variance_reduction": 0.20},
    2000: {"label": "Simulation tools",  "window_correction": 0.42, "sc_awareness": 0.55, "variance_reduction": 0.38},
    2006: {"label": "Full analytics",    "window_correction": 0.62, "sc_awareness": 0.72, "variance_reduction": 0.55},
    2012: {"label": "Predictive models", "window_correction": 0.78, "sc_awareness": 0.85, "variance_reduction": 0.70},
    2018: {"label": "AI / Monte Carlo",  "window_correction": 0.91, "sc_awareness": 0.95, "variance_reduction": 0.88},
}

def get_software_era(year):
    applicable = {y: v for y, v in SOFTWARE_ERAS.items() if y <= year}
    era = applicable[max(applicable.keys())]
    era["year"] = max(applicable.keys())
    return era


# ── Caching ───────────────────────────────────────────────────────────────────

os.makedirs(CACHE_DIR, exist_ok=True)

def cached_get(url, params=None):
    """Fetch from Jolpica with disk caching and automatic pagination."""
    safe = url.replace("/","_").replace(":","").replace("?","_")
    if params:
        safe += "_" + "_".join(f"{k}{v}" for k,v in sorted(params.items()))
    cache_path = os.path.join(CACHE_DIR, safe[:180] + ".json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    # Paginate — Jolpica max 100 per page
    all_races = []
    offset = 0
    limit = 100
    total = None
    base_data = None

    while True:
        p = dict(params or {})
        p["limit"]  = limit
        p["offset"] = offset
        print(f"  → fetching {url} (offset={offset})")
        time.sleep(0.4)
        r = requests.get(url, params=p, timeout=15)
        r.raise_for_status()
        data = r.json()

        if base_data is None:
            base_data = data

        # Dig into whatever table is present and collect its list
        mr = data["MRData"]
        table_key = next(k for k in mr if k.endswith("Table"))
        table = mr[table_key]
        inner_key = next(k for k in table if isinstance(table[k], list))
        page_items = table[inner_key]
        all_races.extend(page_items)

        if total is None:
            total = int(mr.get("total", len(page_items)))
        offset += limit
        if offset >= total:
            break

    # Put all items back into the structure
    mr = base_data["MRData"]
    table_key = next(k for k in mr if k.endswith("Table"))
    inner_key = next(k for k in base_data["MRData"][table_key] if isinstance(base_data["MRData"][table_key][k], list))
    base_data["MRData"][table_key][inner_key] = all_races

    with open(cache_path, "w") as f:
        json.dump(base_data, f)
    return base_data

# ── Ergast fetchers ───────────────────────────────────────────────────────────

def fetch_season_results(year):
    """Fetch all race results for a season."""
    url = f"{ERGAST_BASE}/{year}/results.json"
    data = cached_get(url, {"limit": 500})
    races = data["MRData"]["RaceTable"]["Races"]
    rows = []
    for race in races:
        for r in race.get("Results", []):
            rows.append({
                "year":        year,
                "round":       int(race["round"]),
                "race_name":   race["raceName"],
                "circuit":     race["Circuit"]["circuitId"],
                "driver":      r["Driver"]["driverId"],
                "driver_name": r["Driver"]["familyName"],
                "constructor": r["Constructor"]["constructorId"],
                "grid":        int(r.get("grid", 0)),
                "position":    int(r["position"]) if r.get("positionText","").isdigit() else None,
                "points":      float(r.get("points", 0)),
                "status":      r.get("status",""),
                "laps":        int(r.get("laps", 0)),
            })
    return pd.DataFrame(rows)


def fetch_championship_standings(year):
    """Fetch final driver standings for a season."""
    url = f"{ERGAST_BASE}/{year}/last/driverStandings.json"
    data = cached_get(url, {"limit": 100})
    lists = data["MRData"]["StandingsTable"]["StandingsLists"]
    if not lists:
        return pd.DataFrame()
    rows = []
    for s in lists[0]["DriverStandings"]:
        rows.append({
            "year":        year,
            "position":    int(s["position"]),
            "driver":      s["Driver"]["driverId"],
            "driver_name": s["Driver"]["familyName"],
            "points":      float(s["points"]),
            "wins":        int(s["wins"]),
        })
    return pd.DataFrame(rows)


def fetch_ergast_pitstops(year, circuit):
    """Fetch pit stop data from Ergast (available from 2012+)."""
    url = f"{ERGAST_BASE}/{year}/{circuit}/pitstops.json"
    try:
        data = cached_get(url, {"limit": 200})
        races = data["MRData"]["RaceTable"]["Races"]
        if not races:
            return pd.DataFrame()
        rows = []
        for ps in races[0].get("PitStops", []):
            rows.append({
                "driver":    ps["driverId"],
                "stop":      int(ps["stop"]),
                "lap":       int(ps["lap"]),
                "duration_s": float(ps["duration"]) if ":" not in ps["duration"]
                              else float(ps["duration"].split(":")[0])*60 + float(ps["duration"].split(":")[1]),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ── Local data loaders ────────────────────────────────────────────────────────

def load_pitstops_csv(path=PITSTOPS_CSV):
    if not os.path.exists(path):
        print(f"  ⚠  {path} not found — skipping local pit data")
        return pd.DataFrame()
    df = pd.read_csv(path)
    def to_s(t):
        try:
            t = str(t)
            if ":" in t:
                p = t.split(":")
                return float(p[0])*60 + float(p[1])
            return float(t)
        except:
            return None
    df["pit_s"] = df["Time"].apply(to_s)
    df = df[df["pit_s"].between(10, 120)]
    df = df[df["Lap"] > 3]   # exclude lap 1-3 retirement stops
    return df


def load_lap_progression(path=LAP_TIMES_XL):
    if not os.path.exists(path):
        print(f"  ⚠  {path} not found — using fallback lap times")
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name="Final Data", header=1)
    df.columns = ["_", "Year", "avg_multiplier", "theoretical_lap_s", "_2"]
    df = df[["Year", "avg_multiplier", "theoretical_lap_s"]].dropna()
    df["Year"] = df["Year"].astype(int)
    return df


# ── Strategy variance analysis ────────────────────────────────────────────────

TYPICAL_RACE_LAPS = {
    1994:62, 1995:62, 1996:62, 1997:62, 1998:62, 1999:62,
    2000:62, 2001:62, 2002:62, 2003:62, 2004:62, 2005:62,
    2006:62, 2007:62, 2008:62, 2009:60, 2010:60,
}

def compute_pit_window_errors(pitstops_df, year):
    """
    For each driver/race in a year, compute how far their actual pit lap
    was from the theoretically optimal even-split window.
    Returns a df with driver, race, window_error_laps, estimated_penalty_s
    """
    year_df = pitstops_df[pitstops_df["Year"] == year].copy()
    if year_df.empty:
        return pd.DataFrame()

    total_laps = TYPICAL_RACE_LAPS.get(year, 62)
    results = []

    for (race, driver), grp in year_df.groupby(["Race", "DriverName"]):
        grp = grp.sort_values("Lap")
        n_stops = len(grp)
        actual_laps = grp["Lap"].tolist()

        # Optimal: evenly spaced through race
        optimal_laps = [int(total_laps * i / (n_stops + 1))
                        for i in range(1, n_stops + 1)]

        errors = [abs(a - o) for a, o in zip(actual_laps, optimal_laps)]
        avg_error = float(np.mean(errors))

        # Penalty: each lap of window error ≈ 0.3s extra race time
        # (conservative linear tyre deg, supported by Heilmeier et al.)
        penalty_s = avg_error * 0.3

        results.append({
            "year": year, "race": race, "driver": driver,
            "n_stops": n_stops,
            "actual_laps": actual_laps,
            "optimal_laps": optimal_laps,
            "avg_window_error_laps": round(avg_error, 2),
            "penalty_s": round(penalty_s, 2),
        })

    return pd.DataFrame(results)


def estimate_positions_gained(penalty_s, lap_s, field_spread_s=1.5):
    """
    Roughly how many positions does `penalty_s` seconds correspond to?
    field_spread_s = avg gap between consecutive cars in the era.
    """
    return penalty_s / field_spread_s


def simulate_corrected_season(race_results, window_errors, year, sw_correction):
    """
    Re-simulate a season applying software correction to pit windows.

    sw_correction: fraction of window error eliminated by software (0–1)

    For each race:
      - Estimate penalty each driver suffered
      - Apply correction → reduced penalty
      - Convert penalty delta to position gains/losses
      - Recompute points
    """
    points_sys = get_points_system(year)
    corrected_races = []

    for round_num in sorted(race_results["round"].unique()):
        race = race_results[race_results["round"] == round_num].copy()
        race_name = race["race_name"].iloc[0]
        circuit   = race["circuit"].iloc[0]

        # Get pit window errors for this race
        race_errors = window_errors[
            (window_errors["year"] == year) &
            (window_errors["race"].str.lower() == circuit.lower())
        ] if not window_errors.empty else pd.DataFrame()

        race = race.copy()
        race["original_position"] = race["position"]
        race["strategy_penalty_s"] = 0.0
        race["corrected_penalty_s"] = 0.0

        if not race_errors.empty:
            for _, row in race_errors.iterrows():
                driver_match = race["driver"].str.lower().str.contains(
                    row["driver"].split()[-1].lower(), na=False
                )
                if driver_match.any():
                    idx = race[driver_match].index[0]
                    penalty = row["penalty_s"]
                    race.loc[idx, "strategy_penalty_s"] = penalty
                    race.loc[idx, "corrected_penalty_s"] = penalty * (1 - sw_correction)

        # Estimate position shifts from corrected penalties
        race["penalty_delta_s"] = race["strategy_penalty_s"] - race["corrected_penalty_s"]

        # Sort by (original_position - estimated position gain)
        # position gain ≈ penalty_delta / 1.5s per position
        race["pos_gain"] = (race["penalty_delta_s"] / 1.5).round().astype(int)
        race["simulated_position"] = (race["original_position"] - race["pos_gain"]).clip(lower=1)

        # Recompute points
        race["simulated_points"] = race["simulated_position"].map(
            lambda p: points_sys.get(p, 0)
        )
        race["original_points"] = race["original_position"].map(
            lambda p: points_sys.get(int(p), 0) if pd.notna(p) else 0
        )

        corrected_races.append(race)

    if not corrected_races:
        return pd.DataFrame(), pd.DataFrame()

    all_races = pd.concat(corrected_races, ignore_index=True)

    # Championship tally
    original_champ = (
        all_races.groupby(["driver", "driver_name"])["original_points"]
        .sum().reset_index()
        .sort_values("original_points", ascending=False)
        .reset_index(drop=True)
    )
    original_champ["original_position"] = original_champ.index + 1

    simulated_champ = (
        all_races.groupby(["driver", "driver_name"])["simulated_points"]
        .sum().reset_index()
        .sort_values("simulated_points", ascending=False)
        .reset_index(drop=True)
    )
    simulated_champ["simulated_position"] = simulated_champ.index + 1

    standings = original_champ.merge(simulated_champ, on=["driver","driver_name"])
    standings["points_delta"] = standings["simulated_points"] - standings["original_points"]
    standings["position_change"] = standings["original_position"] - standings["simulated_position"]

    return standings, all_races


# ── Plotting ──────────────────────────────────────────────────────────────────

def setup_fig_style():
    plt.rcParams.update({
        "font.family":       "monospace",
        "axes.facecolor":    COLOUR["card"],
        "figure.facecolor":  COLOUR["bg"],
        "axes.edgecolor":    COLOUR["border"],
        "axes.labelcolor":   COLOUR["muted"],
        "xtick.color":       COLOUR["muted"],
        "ytick.color":       COLOUR["muted"],
        "grid.color":        COLOUR["border"],
        "grid.linewidth":    0.5,
        "text.color":        COLOUR["text"],
        "axes.titlecolor":   COLOUR["text"],
        "axes.titlesize":    11,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
    })


def plot_single_season(year, standings, all_races, window_errors, sw_era):
    setup_fig_style()
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(COLOUR["bg"])

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Title
    fig.text(0.02, 0.97,
             f"F1 {year} — STRATEGY VARIANCE ANALYSIS",
             fontsize=18, fontweight="bold", color=COLOUR["text"],
             fontfamily="monospace", va="top")
    fig.text(0.02, 0.935,
             f"Software era: {sw_era['label']}  ·  "
             f"Window correction: {sw_era['window_correction']*100:.0f}%  ·  "
             f"SC awareness: {sw_era['sc_awareness']*100:.0f}%",
             fontsize=9, color=COLOUR["muted"], fontfamily="monospace", va="top")

    # ── 1. Championship standings comparison ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    if not standings.empty:
        top10 = standings.head(10)
        names = [n[:14] for n in top10["driver_name"]]
        x = np.arange(len(names))
        w = 0.35
        bars1 = ax1.bar(x - w/2, top10["original_points"], w,
                        color=COLOUR["red"], alpha=0.8, label="Historical")
        bars2 = ax1.bar(x + w/2, top10["simulated_points"], w,
                        color=COLOUR["green"], alpha=0.8, label="Software-aided")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=35, ha="right")
        ax1.set_ylabel("Championship Points")
        ax1.set_title(f"Championship Points — Top 10  (Historical vs Software-Aided)")
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.3)

        # Annotate champion
        if not top10.empty:
            champ_hist = top10.iloc[0]
            champ_sw   = standings.sort_values("simulated_points", ascending=False).iloc[0]
            outcome = "SAME CHAMPION" if champ_hist["driver"] == champ_sw["driver"] \
                      else f"CHAMPION CHANGES → {champ_sw['driver_name'].upper()}"
            colour = COLOUR["green"] if champ_hist["driver"] == champ_sw["driver"] else COLOUR["amber"]
            ax1.set_title(
                f"Championship Points — {outcome}",
                color=colour, fontsize=10
            )

    # ── 2. Points delta ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    if not standings.empty:
        top8 = standings.head(8)
        deltas = top8["points_delta"]
        colours = [COLOUR["green"] if d >= 0 else COLOUR["red"] for d in deltas]
        ax2.barh([n[:12] for n in top8["driver_name"]], deltas,
                 color=colours, alpha=0.85)
        ax2.axvline(0, color=COLOUR["border"], linewidth=1)
        ax2.set_xlabel("Points gained with software")
        ax2.set_title("Points Delta (SW − Historical)")
        ax2.grid(axis="x", alpha=0.3)

    # ── 3. Pit window error distribution ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if not window_errors.empty:
        errors = window_errors[window_errors["year"] == year]["avg_window_error_laps"]
        ax3.hist(errors, bins=20, color=COLOUR["blue"], alpha=0.8, edgecolor=COLOUR["border"])
        ax3.axvline(errors.mean(), color=COLOUR["amber"], linestyle="--",
                    linewidth=1.5, label=f"Mean: {errors.mean():.1f} laps")
        ax3.set_xlabel("Window error (laps from optimal)")
        ax3.set_ylabel("Driver-races")
        ax3.set_title("Pit Window Error Distribution")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

    # ── 4. Penalty by race ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if not window_errors.empty:
        yr = window_errors[window_errors["year"] == year]
        race_pen = yr.groupby("race")["penalty_s"].mean().sort_values(ascending=False).head(12)
        ax4.bar(range(len(race_pen)), race_pen.values,
                color=COLOUR["amber"], alpha=0.8)
        ax4.set_xticks(range(len(race_pen)))
        ax4.set_xticklabels([r[:8] for r in race_pen.index], rotation=45, ha="right")
        ax4.set_ylabel("Avg penalty (s)")
        ax4.set_title("Avg Strategy Penalty by Race")
        ax4.grid(axis="y", alpha=0.3)

    # ── 5. Strategy variance vs software era timeline ─────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    eras = sorted(SOFTWARE_ERAS.keys())
    corrections = [SOFTWARE_ERAS[y]["window_correction"] for y in eras]
    variances   = [1 - SOFTWARE_ERAS[y]["variance_reduction"] for y in eras]
    ax5.plot(eras, corrections, color=COLOUR["green"], linewidth=2,
             marker="o", markersize=4, label="Window correction %")
    ax5.plot(eras, variances,   color=COLOUR["red"],   linewidth=2,
             marker="s", markersize=4, label="Remaining variance")
    ax5.axvline(year, color=COLOUR["amber"], linestyle="--",
                linewidth=1.5, label=f"Selected: {year}")
    ax5.set_xlabel("Year")
    ax5.set_ylabel("Fraction (0–1)")
    ax5.set_title("Software Capability Timeline")
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.3)
    ax5.set_ylim(0, 1.05)

    # ── 6. Cumulative points gap (champion vs P2) ─────────────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    if not all_races.empty and not standings.empty:
        champ_driver = standings.iloc[0]["driver"]
        p2_driver    = standings.iloc[1]["driver"] if len(standings) > 1 else None

        for drv, col, label in [
            (champ_driver, COLOUR["amber"], standings.iloc[0]["driver_name"]),
            (p2_driver,    COLOUR["blue"],  standings.iloc[1]["driver_name"] if p2_driver else "P2"),
        ]:
            if drv is None:
                continue
            drv_races = all_races[all_races["driver"] == drv].sort_values("round")
            cum_orig = drv_races["original_points"].cumsum().values
            cum_sim  = drv_races["simulated_points"].cumsum().values
            rounds   = drv_races["round"].values
            ax6.plot(rounds, cum_orig, color=col, linewidth=1.5,
                     linestyle="--", alpha=0.6, label=f"{label} (hist)")
            ax6.plot(rounds, cum_sim,  color=col, linewidth=2.0,
                     label=f"{label} (sw)")

        ax6.set_xlabel("Race round")
        ax6.set_ylabel("Cumulative points")
        ax6.set_title("Cumulative Points — Champion vs P2  (dashed=historical, solid=software-aided)")
        ax6.legend(fontsize=8, ncol=2)
        ax6.grid(alpha=0.3)

    # ── 7. Key findings text ──────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor(COLOUR["card"])
    ax7.axis("off")

    findings = []
    if not standings.empty:
        champ = standings.iloc[0]
        p2    = standings.iloc[1] if len(standings) > 1 else None
        findings.append(f"Champion: {champ['driver_name']}")
        findings.append(f"  Historical pts: {champ['original_points']:.0f}")
        findings.append(f"  SW-aided pts:   {champ['simulated_points']:.0f}")
        if p2 is not None:
            margin_hist = champ["original_points"] - p2["original_points"]
            margin_sw   = champ["simulated_points"] - p2["simulated_points"]
            findings.append(f"  Margin (hist):  +{margin_hist:.0f} pts")
            findings.append(f"  Margin (SW):    +{margin_sw:.0f} pts")
            if abs(margin_hist) <= 10:
                findings.append(f"  ⚠ CLOSE SEASON")
            champ_sw = standings.sort_values("simulated_points", ascending=False).iloc[0]
            if champ_sw["driver"] != champ["driver"]:
                findings.append(f"  ★ OUTCOME FLIPS!")
                findings.append(f"  → {champ_sw['driver_name']} wins")
    if not window_errors.empty:
        yr = window_errors[window_errors["year"] == year]
        findings.append("")
        findings.append(f"Pit window data:")
        findings.append(f"  Races analysed: {yr['race'].nunique()}")
        findings.append(f"  Driver-races:   {len(yr)}")
        findings.append(f"  Avg window err: {yr['avg_window_error_laps'].mean():.1f} laps")
        findings.append(f"  Avg penalty:    {yr['penalty_s'].mean():.2f}s")
        findings.append(f"  SW saves avg:   {yr['penalty_s'].mean()*sw_era['window_correction']:.2f}s")

    ax7.text(0.05, 0.95, "\n".join(findings),
             transform=ax7.transAxes,
             fontsize=8, fontfamily="monospace",
             color=COLOUR["text"], va="top", linespacing=1.6)
    ax7.set_title("Key Findings", fontsize=9, color=COLOUR["red"])

    plt.savefig(f"f1_strategy_{year}.png", dpi=150, bbox_inches="tight",
                facecolor=COLOUR["bg"])
    print(f"  ✓ Saved f1_strategy_{year}.png")
    plt.show()


def plot_era_sweep(sweep_results, pitstops_df):
    """Multi-year overview: variance reduction across eras."""
    setup_fig_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(COLOUR["bg"])
    fig.suptitle("F1 STRATEGY VARIANCE ACROSS ERAS (1994–2010)",
                 fontsize=15, fontweight="bold", color=COLOUR["text"],
                 fontfamily="monospace")

    years = sorted(sweep_results.keys())

    # ── 1. Avg pit window error by year ───────────────────────────────────────
    ax = axes[0, 0]
    yearly_errors = pitstops_df[pitstops_df["Lap"] > 3].copy()

    def to_s(t):
        try:
            t = str(t)
            if ":" in t: p = t.split(":"); return float(p[0])*60+float(p[1])
            return float(t)
        except: return None

    yearly_errors["pit_s"] = yearly_errors["Time"].apply(to_s)
    yearly_errors = yearly_errors[yearly_errors["pit_s"].between(10, 120)]
    yr_mean_err = []
    for yr in years:
        ye = yearly_errors[yearly_errors["Year"] == yr]
        if ye.empty:
            yr_mean_err.append(0)
            continue
        total_laps = TYPICAL_RACE_LAPS.get(yr, 62)
        errors = []
        for (race, driver), grp in ye.groupby(["Race","DriverName"]):
            grp = grp.sort_values("Lap")
            n = len(grp)
            actual   = grp["Lap"].tolist()
            optimal  = [int(total_laps * i / (n+1)) for i in range(1, n+1)]
            errors.append(np.mean([abs(a-o) for a,o in zip(actual,optimal)]))
        yr_mean_err.append(np.mean(errors))

    ax.bar(years, yr_mean_err, color=COLOUR["blue"], alpha=0.8,
           edgecolor=COLOUR["border"])
    sw_correction_line = [SOFTWARE_ERAS[
        max(y for y in SOFTWARE_ERAS if y <= yr)
    ]["window_correction"] * max(yr_mean_err) for yr in years]
    ax.plot(years, [e * (1 - SOFTWARE_ERAS[max(y for y in SOFTWARE_ERAS if y <= yr)]["window_correction"])
                    for yr, e in zip(years, yr_mean_err)],
            color=COLOUR["green"], linewidth=2, marker="o", markersize=4,
            label="After SW correction")
    ax.set_title("Average Pit Window Error by Year")
    ax.set_xlabel("Year"); ax.set_ylabel("Avg laps from optimal")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # ── 2. Championship margins ───────────────────────────────────────────────
    ax = axes[0, 1]
    margins_hist = []
    margins_sw   = []
    for yr in years:
        if yr not in sweep_results:
            margins_hist.append(0); margins_sw.append(0); continue
        st = sweep_results[yr]["standings"]
        if st.empty or len(st) < 2:
            margins_hist.append(0); margins_sw.append(0); continue
        margins_hist.append(st.iloc[0]["original_points"] - st.iloc[1]["original_points"])
        sw_sorted = st.sort_values("simulated_points", ascending=False)
        margins_sw.append(sw_sorted.iloc[0]["simulated_points"] - sw_sorted.iloc[1]["simulated_points"])

    x = np.arange(len(years))
    w = 0.35
    ax.bar(x - w/2, margins_hist, w, color=COLOUR["red"],   alpha=0.8, label="Historical")
    ax.bar(x + w/2, margins_sw,   w, color=COLOUR["green"], alpha=0.8, label="Software-aided")
    ax.set_xticks(x); ax.set_xticklabels(years, rotation=45)
    ax.set_title("Championship Winning Margin")
    ax.set_xlabel("Year"); ax.set_ylabel("Points gap P1–P2")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color=COLOUR["border"])

    # ── 3. Outcome flips ──────────────────────────────────────────────────────
    ax = axes[1, 0]
    flip_data = []
    for yr in years:
        if yr not in sweep_results:
            continue
        st = sweep_results[yr]["standings"]
        if st.empty:
            continue
        hist_champ = st.iloc[0]["driver"]
        sw_champ   = st.sort_values("simulated_points", ascending=False).iloc[0]["driver"]
        margin     = abs(st.iloc[0]["original_points"] - (st.iloc[1]["original_points"] if len(st) > 1 else 0))
        flip_data.append({
            "year": yr, "flips": hist_champ != sw_champ,
            "margin": margin,
            "hist_champ": st.iloc[0]["driver_name"],
            "sw_champ": st.sort_values("simulated_points", ascending=False).iloc[0]["driver_name"],
        })
    fd = pd.DataFrame(flip_data)
    colours = [COLOUR["red"] if f else COLOUR["green"] for f in fd["flips"]]
    ax.bar(fd["year"], fd["margin"], color=colours, alpha=0.85, edgecolor=COLOUR["border"])
    ax.set_title("Championship Margin  (RED = outcome flips with software)")
    ax.set_xlabel("Year"); ax.set_ylabel("Winning margin (pts)")
    ax.grid(axis="y", alpha=0.3)
    # Annotate flips
    for _, row in fd[fd["flips"]].iterrows():
        ax.annotate(f"→{row['sw_champ'][:8]}",
                    xy=(row["year"], row["margin"]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=7, color=COLOUR["amber"])

    # ── 4. Variance reduction vs software capability ──────────────────────────
    ax = axes[1, 1]
    sw_years = sorted(SOFTWARE_ERAS.keys())
    sw_caps  = [SOFTWARE_ERAS[y]["window_correction"] for y in sw_years]
    ax.fill_between(sw_years, sw_caps, alpha=0.2, color=COLOUR["green"])
    ax.plot(sw_years, sw_caps, color=COLOUR["green"], linewidth=2.5,
            marker="o", markersize=5, label="Pit window correction")
    ax.plot(sw_years, [SOFTWARE_ERAS[y]["sc_awareness"] for y in sw_years],
            color=COLOUR["amber"], linewidth=2, linestyle="--",
            marker="s", markersize=4, label="Safety car awareness")
    ax.plot(sw_years, [SOFTWARE_ERAS[y]["variance_reduction"] for y in sw_years],
            color=COLOUR["blue"], linewidth=2, linestyle=":",
            marker="^", markersize=4, label="Overall variance reduction")

    # Shade eras
    era_labels = [
        (1950, 1983, "Pre-telemetry"),
        (1984, 1999, "Telemetry era"),
        (2000, 2011, "Simulation era"),
        (2012, 2024, "AI/ML era"),
    ]
    for start, end, label in era_labels:
        ax.axvspan(start, end, alpha=0.04, color=COLOUR["blue"])
        ax.text((start+end)/2, 0.02, label, ha="center", fontsize=7,
                color=COLOUR["muted"], rotation=0)

    ax.set_xlabel("Year"); ax.set_ylabel("Software capability (0–1)")
    ax.set_title("Software Capability Growth by Dimension")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("f1_era_sweep.png", dpi=150, bbox_inches="tight",
                facecolor=COLOUR["bg"])
    print("  ✓ Saved f1_era_sweep.png")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def analyse_year(year, pitstops_df, lap_df):
    print(f"\n{'='*60}")
    print(f"  Analysing {year}")
    print(f"{'='*60}")

    sw_era = get_software_era(year)
    print(f"  Software era: {sw_era['label']}")
    print(f"  Fetching race results from Ergast...")

    race_results = fetch_season_results(year)
    if race_results.empty:
        print(f"  ✗ No race data for {year}")
        return None, None, None

    print(f"  Fetching championship standings...")
    champ = fetch_championship_standings(year)

    print(f"  Computing pit window errors...")
    window_errors = compute_pit_window_errors(pitstops_df, year)

    print(f"  Running corrected season simulation...")
    standings, all_races = simulate_corrected_season(
        race_results, window_errors, year, sw_era["window_correction"]
    )

    if not standings.empty:
        print(f"\n  Top 5 standings comparison:")
        print(f"  {'Driver':<20} {'Hist pts':>8} {'SW pts':>8} {'Delta':>6} {'Pos Δ':>6}")
        print(f"  {'-'*52}")
        for _, row in standings.head(5).iterrows():
            print(f"  {row['driver_name']:<20} "
                  f"{row['original_points']:>8.0f} "
                  f"{row['simulated_points']:>8.0f} "
                  f"{row['points_delta']:>+6.0f} "
                  f"{row['position_change']:>+6.0f}")

        hist_champ = standings.iloc[0]["driver_name"]
        sw_champ   = standings.sort_values("simulated_points", ascending=False).iloc[0]["driver_name"]
        if hist_champ != sw_champ:
            print(f"\n  ★ CHAMPIONSHIP OUTCOME CHANGES!")
            print(f"    Historical champion: {hist_champ}")
            print(f"    Software-aided champion: {sw_champ}")
        else:
            print(f"\n  ✓ Same champion: {hist_champ}")

    return standings, all_races, window_errors


def main():
    parser = argparse.ArgumentParser(
        description="F1 Strategy Variance Analyser — LUT University Thesis Tool"
    )
    parser.add_argument("--year",  type=int, default=2010,
                        help="Single season to analyse (default: 2010)")
    parser.add_argument("--mode",  choices=["single","sweep"], default="single",
                        help="single = one year, sweep = 1994–2010")
    parser.add_argument("--start", type=int, default=1994)
    parser.add_argument("--end",   type=int, default=2010)
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip chart rendering (just print results)")
    args = parser.parse_args()

    print("\n  F1 Strategy Variance Analyser")
    print("  LUT University — Bachelor's Thesis\n")

    # Load local data
    print("Loading local data files...")
    pitstops_df = load_pitstops_csv(PITSTOPS_CSV)
    lap_df      = load_lap_progression(LAP_TIMES_XL)
    print(f"  Pit stops: {len(pitstops_df)} records ({pitstops_df['Year'].min() if not pitstops_df.empty else 'N/A'}–{pitstops_df['Year'].max() if not pitstops_df.empty else 'N/A'})")
    print(f"  Lap times: {len(lap_df)} years")

    if args.mode == "single":
        standings, all_races, window_errors = analyse_year(
            args.year, pitstops_df, lap_df
        )
        if standings is not None and not args.no_plot:
            sw_era = get_software_era(args.year)
            plot_single_season(args.year, standings, all_races, window_errors, sw_era)

    elif args.mode == "sweep":
        sweep_results = {}
        for year in range(args.start, args.end + 1):
            standings, all_races, window_errors = analyse_year(
                year, pitstops_df, lap_df
            )
            if standings is not None:
                sweep_results[year] = {
                    "standings":     standings,
                    "all_races":     all_races,
                    "window_errors": window_errors,
                }

        if not args.no_plot:
            plot_era_sweep(sweep_results, pitstops_df)

        # Summary CSV export
        rows = []
        for yr, res in sweep_results.items():
            st = res["standings"]
            if st.empty:
                continue
            hist_champ = st.iloc[0]
            sw_champ   = st.sort_values("simulated_points", ascending=False).iloc[0]
            rows.append({
                "year":               yr,
                "hist_champion":      hist_champ["driver_name"],
                "hist_champion_pts":  hist_champ["original_points"],
                "sw_champion":        sw_champ["driver_name"],
                "sw_champion_pts":    sw_champ["simulated_points"],
                "outcome_flips":      hist_champ["driver"] != sw_champ["driver"],
                "sw_era_label":       get_software_era(yr)["label"],
                "sw_correction":      get_software_era(yr)["window_correction"],
            })
        if rows:
            out = pd.DataFrame(rows)
            out.to_csv("f1_sweep_summary.csv", index=False)
            print("\n  ✓ Saved f1_sweep_summary.csv")
            print("\n  SWEEP SUMMARY:")
            print(out.to_string(index=False))


if __name__ == "__main__":
    main()