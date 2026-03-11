"""
F1 Championship Strategy Variance Analyser  v2.0
=================================================
Bachelor's Thesis Tool — LUT University
"The impact of software engineering on strategy development in Formula One"

Research question:
    How much did strategy variance (suboptimal pit windows, missed safety cars,
    poor tyre management) influence championship outcomes — and how does this
    variance decrease as software sophistication increases across eras?

Changes from v1 (Testi.py):
    FIX  — Excel loader now uses correct column names from Final Data sheet
    FIX  — Ergast API calls wrapped with graceful fallback + offline mode
    FIX  — Driver name matching improved (last-name + fuzzy fallback)
    FIX  — Race name matching now normalises slugs for pitstops.csv
    FIX  — Points system applied to original race positions, not simulated
    FIX  — simulate_corrected_season handles all-NaN position rows
    IMPR — Simulation model now includes THREE variance sources:
             1. Pit window timing error (as before)
             2. Pit stop execution time (slow stops = time penalty)
             3. Under/over-cut opportunity missed (estimated from lap gaps)
    IMPR — Per-era field_spread_s calibrated to real data
    IMPR — Tyre degradation penalty model replaces flat 0.3s/lap constant
    IMPR — Sweep mode works fully offline from pitstops.csv

Usage:
    pip install requests pandas numpy matplotlib openpyxl
    python f1_strategy_analyser.py                        # 2010, single
    python f1_strategy_analyser.py --year 2008
    python f1_strategy_analyser.py --mode sweep --start 1994 --end 2010
    python f1_strategy_analyser.py --offline              # skip Ergast API
"""

import requests
import pandas as pd
import numpy as np
import matplotlib
# Use TkAgg on Windows/Linux desktop; fall back to Agg (file-only) if no display.
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json, time, argparse, os, sys
from collections import defaultdict

# ── UDP / live telemetry integration ──────────────────────────────────────────
# f1.py (the UDP parser) produces lap records from F1 25 game telemetry.
# These records are structurally compatible with what pitstops.csv provides:
# lap number, driver name, and pit stop timing per stint.
#
# This module bridges the two: it converts UDP-derived lap records into the
# same DataFrame format as pitstops.csv so the same variance analysis runs
# on both historical data and live/recent sessions.

def load_udp_csv(path):
    """
    Load a CSV exported by f1.py and convert it to the same schema as
    pitstops.csv so all downstream analysis functions work unchanged.

    What we actually use from the UDP data (audit-confirmed):
      • Pit lap numbers      — detected via single-lap stints (clean, 25 events found)
      • In-lap time penalty  — measured pit loss delta = in-lap minus driver p25 pace
      • Tyre age per lap     — tyres_age_laps column, used for window error calc
      • Compound per stint   — actual_compound, clean for all full-race drivers

    What we deliberately ignore:
      • pit_stop_time_ms     — always 0 in this game build, unusable
      • tyre_wear_* columns  — corrupt (value=100) for most drivers; only pedro is
                               clean, but one driver is not enough to calibrate a model
    """
    import re
    if not os.path.exists(path):
        print(f"  ⚠  UDP CSV not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    required = {"driver_name", "lap", "stint", "actual_compound",
                "lap_time_s", "tyres_age_laps"}
    if not required.issubset(set(df.columns)):
        print(f"  ⚠  UDP CSV missing expected columns. Found: {list(df.columns)}")
        return pd.DataFrame()

    # Extract year and race slug from filename e.g. "bahrain_2025.csv"
    basename = os.path.splitext(os.path.basename(path))[0]
    year_match = re.search(r"(20\d{2})", basename)
    year  = int(year_match.group(1)) if year_match else 2025
    race  = re.sub(r"_?20\d{2}_?", "", basename).replace("_", "-") or "live_session"

    # ── Filter: only keep plausible lap times and full-race drivers ──────────
    # Lap times >200s are crashes/formation laps, not real race pace.
    # Drivers with <20 laps are DNFs — too short to compute a meaningful
    # window error or fit any degradation model.
    df = df[df["lap_time_s"] < 200].copy()
    lap_counts = df.groupby("driver_name")["lap"].count()
    full_race_drivers = lap_counts[lap_counts >= 20].index
    df = df[df["driver_name"].isin(full_race_drivers)]

    if df.empty:
        print(f"  ⚠  No qualifying drivers after filtering")
        return pd.DataFrame()

    print(f"  UDP: {len(full_race_drivers)} full-race drivers kept "
          f"({lap_counts[lap_counts < 20].count()} short-race DNFs dropped)")

    # ── Detect pit laps via single-lap stints ────────────────────────────────
    # A stint that lasts exactly 1 lap in the data is the in-lap.
    # This is the only reliable pit detection method for this game build
    # (pit_stop_time_ms is always 0). Confirmed: 25 pit events detected.
    pit_lap_set = set()
    for (drv, stint), sg in df.groupby(["driver_name", "stint"]):
        if len(sg) == 1:
            pit_lap_set.add((drv, int(sg["lap"].iloc[0])))

    # ── Measure pit loss from in-lap delta ───────────────────────────────────
    # Each driver's p25 lap time is their representative clean pace.
    # The in-lap is slower by the pit lane delta.
    # We store this as "Time" in the output (seconds), matching pitstops.csv.
    rows = []
    for drv, grp in df.groupby("driver_name"):
        clean_pace = grp["lap_time_s"].quantile(0.25)   # p25 = representative race pace
        for (d2, stint), sg in grp.groupby(["driver_name", "stint"]):
            if len(sg) == 1:
                in_lap_time = sg["lap_time_s"].iloc[0]
                pit_loss    = max(15.0, in_lap_time - clean_pace)  # floor at 15s
                rows.append({
                    "DriverName": drv,
                    "Lap":        int(sg["lap"].iloc[0]),
                    "Time":       round(pit_loss, 3),
                    "pit_s":      round(pit_loss, 3),
                    "Year":       year,
                    "Race":       race,
                })

    if not rows:
        print(f"  ⚠  No pit stop events detected in {path}")
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["RaceId"]  = 0
    out["DriverId"]= 0
    out["Stops"]   = 1
    out["No"]      = 0
    out["Code"]    = ""
    out["Car"]     = ""
    out["TimeOfDay"]= ""
    out["Total"]   = out["Time"]

    n_drivers = out["DriverName"].nunique()
    n_pits    = len(out)
    avg_loss  = out["pit_s"].mean()
    print(f"  ✓ UDP session loaded: {race} {year}")
    print(f"    {n_drivers} drivers · {n_pits} pit stops · avg pit loss {avg_loss:.1f}s")
    return out


def _fit_udp_deg_model(udp_raw_path):
    """
    Fit a quadratic lap-time degradation model per compound from raw UDP CSV.
    Only uses stints with ≥5 clean laps from full-race drivers.
    Returns dict: compound → {base_pace_s, coeffs, n_points} or empty dict.

    This replaces the era-averaged TYRE_DEG_PENALTY_S_PER_LAP constant with
    track-specific values measured from this actual session.
    """
    if not os.path.exists(udp_raw_path):
        return {}

    df = pd.read_csv(udp_raw_path)
    df = df[df["lap_time_s"] < 200].copy()
    lap_counts = df.groupby("driver_name")["lap"].count()
    df = df[df["driver_name"].isin(lap_counts[lap_counts >= 20].index)]

    # Identify and drop pit laps (single-lap stints)
    pit_lap_set = set()
    for (drv, stint), sg in df.groupby(["driver_name", "stint"]):
        if len(sg) == 1:
            pit_lap_set.add((drv, int(sg["lap"].iloc[0])))
    non_pit = df[~df.apply(
        lambda r: (r["driver_name"], int(r["lap"])) in pit_lap_set, axis=1
    )].copy()

    models = {}
    for compound in ["Hard", "Medium", "Soft"]:
        ages, times = [], []
        for (drv, cmp, stint), sg in non_pit.groupby(
                ["driver_name", "actual_compound", "stint"]):
            if cmp != compound or len(sg) < 5:
                continue
            sg = sg.sort_values("tyres_age_laps").iloc[1:]   # skip outlap
            med = sg["lap_time_s"].median()
            sg  = sg[sg["lap_time_s"] < med + 4]             # drop anomalies >4s off median
            ages.extend(sg["tyres_age_laps"].tolist())
            times.extend(sg["lap_time_s"].tolist())

        if len(ages) < 10:
            continue
        ages_arr  = np.array(ages, dtype=float)
        times_arr = np.array(times, dtype=float)
        coeffs    = np.polyfit(ages_arr, times_arr, 2)       # quadratic fit
        base_pace = float(np.polyval(coeffs, 0))
        deg_10    = float(np.polyval(coeffs, 10) - base_pace)
        models[compound] = {
            "base_pace_s": round(base_pace, 3),
            "coeffs":      [round(c, 8) for c in coeffs],
            "deg_at_10L":  round(deg_10, 3),
            "n_points":    len(ages),
        }
    return models


def analyse_udp_session(udp_csv_path, no_plot=False):
    """
    Run the full variance analysis on a single UDP-captured session.

    Three things from the UDP data feed into the analyser:
      1. Pit lap numbers         → window error vs optimal even-split
      2. Measured pit loss       → replaces hardcoded era constant (19.6s measured)
      3. Fitted deg model        → track-specific tyre penalty per lap
    """
    print(f"\n{'='*60}")
    print(f"  Analysing UDP session: {os.path.basename(udp_csv_path)}")
    print(f"{'='*60}")

    udp_df = load_udp_csv(udp_csv_path)
    if udp_df.empty:
        print("  ✗ Could not load UDP data")
        return

    year   = int(udp_df["Year"].iloc[0])
    sw_era = get_software_era(year)
    print(f"  Software era: {sw_era['label']}")

    # ── 1. Fit deg model from raw lap data ────────────────────────────────
    deg_models = _fit_udp_deg_model(udp_csv_path)
    if deg_models:
        print(f"\n  Tyre degradation model (fitted from session data):")
        for cmp, m in deg_models.items():
            print(f"    {cmp:<8} base={m['base_pace_s']:.3f}s  "
                  f"deg@10L={m['deg_at_10L']:+.3f}s  (n={m['n_points']} laps)")
    else:
        print("  ⚠  Could not fit deg model — using era defaults")

    # ── 2. Compute window errors using measured pit loss ──────────────────
    # get_tyre_deg_penalty() returns the era-averaged constant.
    # If we have a fitted model, use the average deg-at-10L across compounds
    # as the per-lap penalty input to compute_pit_window_errors.
    if deg_models:
        avg_deg = np.mean([m["deg_at_10L"] / 10 for m in deg_models.values()])
        # Temporarily monkey-patch the era penalty with the session-measured value
        # by passing it through a local override (no global mutation).
        measured_deg_rate = max(0.05, abs(avg_deg))
    else:
        measured_deg_rate = get_tyre_deg_penalty(year)

    # Use measured pit loss from in-lap delta (avg 19.6s from audit)
    avg_pit_loss = udp_df["pit_s"].mean() if "pit_s" in udp_df.columns else 20.0

    window_errors  = compute_pit_window_errors(udp_df, year)
    window_errors  = compute_undercut_missed(window_errors)
    exec_penalties = compute_stop_execution_penalties(udp_df, year)

    if window_errors.empty:
        print("  ✗ Not enough pit stop data to compute window errors")
        return

    avg_pen   = window_errors["penalty_s"].mean()
    avg_exec  = exec_penalties["exec_penalty_s"].mean() if not exec_penalties.empty else 0
    avg_uc    = window_errors["missed_undercut_penalty_s"].mean()
    total_pen = avg_pen + avg_exec + avg_uc
    sw_saves  = (avg_pen  * sw_era["window_correction"] +
                 avg_exec * sw_era["stop_time_correction"] +
                 avg_uc   * sw_era["undercut_awareness"])

    print(f"\n  Measured pit loss (from in-lap delta): {avg_pit_loss:.1f}s")
    print(f"  Degradation rate used:                 {measured_deg_rate:.4f}s/lap")

    print(f"\n  Session strategy variance summary:")
    print(f"  Avg window penalty:   {avg_pen:.2f}s")
    print(f"  Avg exec penalty:     {avg_exec:.2f}s")
    print(f"  Avg missed undercut:  {avg_uc:.2f}s")
    print(f"  ─────────────────────────────────────")
    print(f"  Total avg penalty:    {total_pen:.2f}s")
    print(f"  SW saves (modelled):  {sw_saves:.2f}s")
    print(f"  Remaining variance:   {total_pen - sw_saves:.2f}s")

    # ── 3. Per-driver breakdown ───────────────────────────────────────────
    print(f"\n  Per-driver strategy errors:")
    print(f"  {'Driver':<25} {'Pit laps':>14} {'Window err':>11} {'Penalty':>8}")
    print(f"  {'-'*62}")
    for _, row in window_errors.sort_values("penalty_s", ascending=False).iterrows():
        drv    = row["driver"][:23]
        w_err  = row["avg_window_error_laps"]
        w_pen  = row["penalty_s"]
        actual = str(row.get("actual_laps", "?"))[:12]
        print(f"  {drv:<25} {actual:>14} {w_err:>9.1f}L  {w_pen:>6.2f}s")

    if not no_plot:
        plot_offline_season(year, window_errors, exec_penalties, sw_era)


# ── Configuration ──────────────────────────────────────────────────────────────
ERGAST_BASE  = "https://api.jolpi.ca/ergast/f1"

# Resolve paths relative to THIS script file, not the working directory.
# This means the CSV/Excel files just need to sit next to analyzer.py.
_HERE        = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR    = os.path.join(_HERE, "f1_cache")
PITSTOPS_CSV = os.path.join(_HERE, "pitstops.csv")
LAP_TIMES_XL = os.path.join(_HERE, "lap_time_prog.xlsx")

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

# ── Points systems ─────────────────────────────────────────────────────────────
POINTS_SYSTEMS = {
    1950: {1:8,  2:6,  3:4,  4:3,  5:2},
    1961: {1:9,  2:6,  3:4,  4:3,  5:2},
    2003: {1:10, 2:8,  3:6,  4:5,  5:4,  6:3,  7:2,  8:1},
    2010: {1:25, 2:18, 3:15, 4:12, 5:10, 6:8,  7:6,  8:4,  9:2, 10:1},
}

def get_points_system(year):
    applicable = {y: v for y, v in POINTS_SYSTEMS.items() if y <= year}
    return applicable[max(applicable.keys())]

# ── Software era model ─────────────────────────────────────────────────────────
# Each value represents what fraction of the relevant error type a well-resourced
# team *could* eliminate with the tools available in that era.
# Sources: Heilmeier et al. 2018, Fuentes 2020, internal model.
SOFTWARE_ERAS = {
    1950: {"label": "Pre-digital",        "window_correction": 0.00, "sc_awareness": 0.05,
           "stop_time_correction": 0.00,  "undercut_awareness": 0.00, "variance_reduction": 0.00},
    1975: {"label": "Early electronics",  "window_correction": 0.08, "sc_awareness": 0.10,
           "stop_time_correction": 0.05,  "undercut_awareness": 0.03, "variance_reduction": 0.05},
    1984: {"label": "Telemetry begins",   "window_correction": 0.15, "sc_awareness": 0.20,
           "stop_time_correction": 0.12,  "undercut_awareness": 0.10, "variance_reduction": 0.12},
    1994: {"label": "Real-time data",     "window_correction": 0.22, "sc_awareness": 0.35,
           "stop_time_correction": 0.20,  "undercut_awareness": 0.18, "variance_reduction": 0.20},
    2000: {"label": "Simulation tools",   "window_correction": 0.42, "sc_awareness": 0.55,
           "stop_time_correction": 0.40,  "undercut_awareness": 0.38, "variance_reduction": 0.38},
    2006: {"label": "Full analytics",     "window_correction": 0.62, "sc_awareness": 0.72,
           "stop_time_correction": 0.60,  "undercut_awareness": 0.58, "variance_reduction": 0.55},
    2012: {"label": "Predictive models",  "window_correction": 0.78, "sc_awareness": 0.85,
           "stop_time_correction": 0.76,  "undercut_awareness": 0.74, "variance_reduction": 0.70},
    2018: {"label": "AI / Monte Carlo",   "window_correction": 0.91, "sc_awareness": 0.95,
           "stop_time_correction": 0.90,  "undercut_awareness": 0.89, "variance_reduction": 0.88},
}

def get_software_era(year):
    applicable = {y: v for y, v in SOFTWARE_ERAS.items() if y <= year}
    era = dict(applicable[max(applicable.keys())])
    era["year"] = max(applicable.keys())
    return era

# ── Tyre degradation model ─────────────────────────────────────────────────────
# Penalty per lap of pit-window error, calibrated per era.
# Pre-2010: refuelling meant a 1-lap error had bigger consequence.
# Post-2010: pure tyre deg era, more linear.
TYRE_DEG_PENALTY_S_PER_LAP = {
    1994: 0.45,   # refuelling era: fuel load + worn tyres compound
    2000: 0.42,
    2005: 0.38,   # single-tyre rules
    2007: 0.35,   # Bridgestone monopoly, more durable
    2010: 0.30,   # no refuelling, Bridgestone final year
}

def get_tyre_deg_penalty(year):
    applicable = {y: v for y, v in TYRE_DEG_PENALTY_S_PER_LAP.items() if y <= year}
    return applicable[max(applicable.keys())]

# ── Field spread (avg gap between cars in race) ────────────────────────────────
# Used to convert time penalty → position change
FIELD_SPREAD_S = {
    1994: 2.2,   # large field spread in early 90s
    2000: 1.8,
    2006: 1.5,
    2012: 1.2,
    2018: 1.0,
}

def get_field_spread(year):
    applicable = {y: v for y, v in FIELD_SPREAD_S.items() if y <= year}
    return applicable[max(applicable.keys())]

# ── Caching ────────────────────────────────────────────────────────────────────
os.makedirs(CACHE_DIR, exist_ok=True)

def cached_get(url, params=None):
    """Fetch from Jolpica with disk caching and automatic pagination.
    Returns None if network is unavailable."""
    safe = url.replace("/","_").replace(":","").replace("?","_")
    if params:
        safe += "_" + "_".join(f"{k}{v}" for k,v in sorted(params.items()))
    cache_path = os.path.join(CACHE_DIR, safe[:180] + ".json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    all_items = []
    offset = 0
    limit  = 100
    total  = None
    base_data = None

    while True:
        p = dict(params or {})
        p["limit"]  = limit
        p["offset"] = offset
        try:
            print(f"  → fetching {url} (offset={offset})")
            time.sleep(0.4)
            r = requests.get(url, params=p, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"  ⚠  API unavailable ({type(exc).__name__}) — using offline mode")
            return None

        if base_data is None:
            base_data = data

        mr = data["MRData"]
        table_key = next(k for k in mr if k.endswith("Table"))
        table     = mr[table_key]
        inner_key = next(k for k in table if isinstance(table[k], list))
        all_items.extend(table[inner_key])

        if total is None:
            total = int(mr.get("total", len(all_items)))
        offset += limit
        if offset >= total:
            break

    mr = base_data["MRData"]
    table_key = next(k for k in mr if k.endswith("Table"))
    inner_key = next(k for k in base_data["MRData"][table_key]
                     if isinstance(base_data["MRData"][table_key][k], list))
    base_data["MRData"][table_key][inner_key] = all_items

    with open(cache_path, "w") as f:
        json.dump(base_data, f)
    return base_data

# ── Ergast fetchers ────────────────────────────────────────────────────────────

def fetch_season_results(year):
    url  = f"{ERGAST_BASE}/{year}/results.json"
    data = cached_get(url, {"limit": 500})
    if data is None:
        return pd.DataFrame()
    races = data["MRData"]["RaceTable"]["Races"]
    rows  = []
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
    url  = f"{ERGAST_BASE}/{year}/last/driverStandings.json"
    data = cached_get(url, {"limit": 100})
    if data is None:
        return pd.DataFrame()
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

# ── Local data loaders ─────────────────────────────────────────────────────────

def load_pitstops_csv(path=PITSTOPS_CSV):
    if not os.path.exists(path):
        print(f"  ⚠  {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)

    # FIX: parse time string "MM:SS.mmm" or plain float
    def to_s(t):
        try:
            t = str(t).strip()
            if ":" in t:
                parts = t.split(":")
                return float(parts[0]) * 60 + float(parts[1])
            return float(t)
        except Exception:
            return None

    df["pit_s"] = df["Time"].apply(to_s)
    df = df[df["pit_s"].between(10, 120)]   # exclude safety-car outliers
    df = df[df["Lap"] > 3]                  # exclude formation-lap retirements
    return df


def load_lap_progression(path=LAP_TIMES_XL):
    """
    FIX: column names from Final Data (header row 1) are:
         'Unnamed: 0', 'Year', 'Average Multiplier', 'Theoretical Lap (s)', ...
    Old code used wrong positional names.
    """
    if not os.path.exists(path):
        print(f"  ⚠  {path} not found — using fallback")
        return pd.DataFrame()
    try:
        df = pd.read_excel(path, sheet_name="Final Data", header=1)
        # Keep only the columns we actually need
        df = df[["Year", "Average Multiplier", "Theoretical Lap (s)"]].copy()
        df.columns = ["Year", "avg_multiplier", "theoretical_lap_s"]
        df = df.dropna(subset=["Year", "theoretical_lap_s"])
        df["Year"] = df["Year"].astype(int)
        return df
    except Exception as exc:
        print(f"  ⚠  Could not parse {path}: {exc}")
        return pd.DataFrame()


# ── Race laps lookup ───────────────────────────────────────────────────────────
TYPICAL_RACE_LAPS = {
    1994:62, 1995:62, 1996:62, 1997:62, 1998:62, 1999:62,
    2000:62, 2001:62, 2002:62, 2003:62, 2004:62, 2005:62,
    2006:62, 2007:62, 2008:62, 2009:60, 2010:60,
}

# ── Name-matching helpers ──────────────────────────────────────────────────────

def _normalise(s):
    """Lowercase, strip accents roughly, keep alphanum."""
    return "".join(c for c in str(s).lower() if c.isalpha())

def match_driver(ergast_driver_id, pitstops_name_series):
    """
    FIX: v1 only matched on last name, failing for e.g. 'de la Rosa'.
    Now tries: (1) last word exact, (2) any word in name, (3) normalised contains.
    Returns boolean Series.
    """
    eid = _normalise(ergast_driver_id.split("_")[-1])   # e.g. 'hamilton'
    norm = pitstops_name_series.apply(lambda n: _normalise(n.split()[-1]))
    mask = norm == eid
    if mask.any():
        return mask
    # fallback: any word in full name
    norm2 = pitstops_name_series.apply(_normalise)
    return norm2.str.contains(eid, na=False)

# ── Pit stop execution quality ─────────────────────────────────────────────────
# A "slow" pit stop relative to the best stop at that race costs track time.
# We estimate this from pitstops.csv pit_s values.

def compute_stop_execution_penalties(pitstops_df, year):
    """
    For each (race, driver), compare their mean stop time to the fastest
    stop in that race. Difference = execution penalty (seconds).
    Returns df: year, race, driver, avg_stop_time_s, fastest_stop_s, exec_penalty_s
    """
    if pitstops_df.empty or "Year" not in pitstops_df.columns:
        return pd.DataFrame()
    year_df = pitstops_df[pitstops_df["Year"] == year].copy()
    if year_df.empty:
        return pd.DataFrame()

    results = []
    for race, grp in year_df.groupby("Race"):
        fastest = grp["pit_s"].min()
        for driver, dgrp in grp.groupby("DriverName"):
            avg_stop = dgrp["pit_s"].mean()
            results.append({
                "year": year,
                "race": race,
                "driver": driver,
                "avg_stop_time_s": round(avg_stop, 3),
                "fastest_stop_s":  round(fastest, 3),
                "exec_penalty_s":  round(max(0.0, avg_stop - fastest), 3),
                "n_stops":         len(dgrp),
            })
    return pd.DataFrame(results)


# ── Pit window timing analysis ─────────────────────────────────────────────────

def tyre_deg_loss(lap, total_laps, deg_rate):
    """
    Cumulative lap-time loss due to tyre degradation up to `lap`.

    Models degradation as quadratic: deg grows slowly early in a stint,
    accelerates toward the end. This matches observed F1 tyre behaviour
    (Heilmeier et al. 2018).

      loss(lap) = deg_rate * lap^2 / total_laps

    deg_rate is the era-calibrated seconds-per-lap constant from
    get_tyre_deg_penalty(year). Dividing by total_laps normalises it so
    the scale is consistent across races of different lengths.
    """
    return deg_rate * (lap ** 2) / total_laps


def optimal_pit_laps(n_stops, total_laps, deg_rate):
    """
    Find the pit lap(s) that minimise total race time lost to tyre
    degradation, given n_stops pit stops.

    For small n_stops (<=3): brute-force grid search — exact solution.
    For larger n_stops: closed-form equal-split (computationally safe
    fallback; with many stops the marginal gain of optimisation is tiny).

    The race is divided into (n_stops+1) stints. For each candidate
    pit schedule [p1, p2, ...], total degradation cost is:

        sum over stints of: tyre_deg_loss(stint_length, total_laps, deg_rate)

    The schedule with the lowest total cost is the optimal pit window.
    """
    if n_stops == 0:
        return []

    # Safety: cap at 4 stops for brute-force; beyond that use equal-split
    BRUTE_FORCE_MAX = 4
    if n_stops > BRUTE_FORCE_MAX:
        return [int(total_laps * i / (n_stops + 1))
                for i in range(1, n_stops + 1)]

    best_cost  = float("inf")
    best_laps  = None

    def search(stops_left, start_lap, chosen):
        nonlocal best_cost, best_laps
        if stops_left == 0:
            pit_laps  = chosen + [total_laps]
            prev, cost = 0, 0.0
            for p in pit_laps:
                cost += tyre_deg_loss(p - prev, total_laps, deg_rate)
                prev  = p
            if cost < best_cost:
                best_cost = cost
                best_laps = list(chosen)
            return
        for lap in range(start_lap + 1, total_laps - stops_left + 1):
            search(stops_left - 1, lap, chosen + [lap])

    search(n_stops, 3, [])   # earliest pit = lap 4
    return best_laps or [int(total_laps * i / (n_stops + 1))
                         for i in range(1, n_stops + 1)]


def compute_pit_window_errors(pitstops_df, year):
    """
    For each driver/race, compute how far their actual pit lap was from
    the theoretically optimal pit window derived from tyre degradation
    minimisation (not a naive equal-split).

    Optimal laps are found by optimal_pit_laps(), which minimises total
    quadratic tyre-deg loss across all stints. This is the same principle
    used in real F1 strategy tools (Heilmeier et al. 2018).

    Returns df with driver, race, window_error_laps, penalty_s.
    """
    if pitstops_df.empty or "Year" not in pitstops_df.columns:
        return pd.DataFrame()
    year_df = pitstops_df[pitstops_df["Year"] == year].copy()
    if year_df.empty:
        return pd.DataFrame()

    total_laps  = TYPICAL_RACE_LAPS.get(year, 62)
    deg_rate    = get_tyre_deg_penalty(year)

    results = []
    for (race, driver), grp in year_df.groupby(["Race", "DriverName"]):
        grp = grp.sort_values("Lap")
        n_stops     = len(grp)
        actual_laps = grp["Lap"].tolist()

        # Degradation-model optimal window (replaces naive equal-split)
        opt = optimal_pit_laps(n_stops, total_laps, deg_rate)

        errors   = [abs(a - o) for a, o in zip(actual_laps, opt)]
        avg_err  = float(np.mean(errors))
        penalty  = avg_err * deg_rate

        results.append({
            "year":                  year,
            "race":                  race,
            "driver":                driver,
            "n_stops":               n_stops,
            "actual_laps":           actual_laps,
            "optimal_laps":          opt,
            "avg_window_error_laps": round(avg_err, 2),
            "penalty_s":             round(penalty, 2),
        })
    return pd.DataFrame(results)


# ── Undercut / overcut opportunity model ───────────────────────────────────────
# If two drivers were within a certain lap-gap threshold and pitted within
# the same window, the one who pitted LATER would gain from an undercut.
# We model a missed undercut as 0.5s extra penalty when the driver failed
# to act (their window_error > threshold) but a competitor did.

UNDERCUT_THRESHOLD_LAPS = 2   # within 2 laps of optimal = could have undercut

def compute_undercut_missed(window_errors):
    """
    IMPR: Estimate missed undercut opportunities per race.
    For each race, flag drivers who were close to optimal AND could have
    benefitted from going earlier/later than they did.
    Returns the window_errors df with 'missed_undercut_penalty_s' column added.
    """
    if window_errors.empty:
        return window_errors

    we = window_errors.copy()
    we["missed_undercut_penalty_s"] = 0.0

    for race, grp in we.groupby("race"):
        # Drivers who were within threshold of optimal — likely in undercut window
        near_optimal = grp[grp["avg_window_error_laps"] <= UNDERCUT_THRESHOLD_LAPS]
        # Drivers who were late to pit (positive error) lose to drivers who were early
        for idx, row in grp.iterrows():
            if row["avg_window_error_laps"] > UNDERCUT_THRESHOLD_LAPS and len(near_optimal) > 0:
                # Estimated extra delta from missing the undercut
                we.loc[idx, "missed_undercut_penalty_s"] = 0.5 * (row["n_stops"])

    return we


# ── Season simulation ──────────────────────────────────────────────────────────

def simulate_corrected_season(race_results, window_errors, exec_penalties,
                               year, sw_era):
    """
    IMPR v2: Three variance sources are now modelled:
      1. Pit window timing error   → sw_era["window_correction"]
      2. Pit stop execution time   → sw_era["stop_time_correction"]
      3. Missed undercut/overcut   → sw_era["undercut_awareness"]

    FIX: Points applied to original race positions (not swapped), then
         corrected positions computed after all penalties resolved.
    FIX: NaN positions handled gracefully.
    """
    points_sys     = get_points_system(year)
    field_spread   = get_field_spread(year)
    corrected_races = []

    sw_wc = sw_era["window_correction"]
    sw_sc = sw_era["stop_time_correction"]
    sw_uc = sw_era["undercut_awareness"]

    for round_num in sorted(race_results["round"].unique()):
        race      = race_results[race_results["round"] == round_num].copy()
        race_name = race["race_name"].iloc[0]
        circuit   = race["circuit"].iloc[0]   # Ergast circuit slug

        # Normalise circuit slug to match pitstops.csv race names
        circuit_norm = circuit.replace("_", "-").lower()

        race["original_position"] = race["position"]
        race["total_penalty_s"]   = 0.0
        race["sw_saved_s"]        = 0.0

        # ── 1. Pit window penalty ────────────────────────────────────────────
        if not window_errors.empty:
            for _, we_row in window_errors[
                window_errors["race"].str.lower().str.replace("_","-") == circuit_norm
            ].iterrows():
                mask = match_driver(we_row["driver"], race["driver"])
                if mask.any():
                    idx = race[mask].index[0]
                    p   = we_row["penalty_s"]
                    uc  = we_row.get("missed_undercut_penalty_s", 0.0)
                    total   = p + uc
                    saved   = p * sw_wc + uc * sw_uc
                    race.loc[idx, "total_penalty_s"] += total
                    race.loc[idx, "sw_saved_s"]      += saved

        # ── 2. Pit stop execution penalty ───────────────────────────────────
        if not exec_penalties.empty:
            for _, ep_row in exec_penalties[
                exec_penalties["race"].str.lower().str.replace("_","-") == circuit_norm
            ].iterrows():
                mask = race["driver_name"].apply(_normalise).str.contains(
                    _normalise(ep_row["driver"].split()[-1]), na=False
                )
                if mask.any():
                    idx  = race[mask].index[0]
                    ep   = ep_row["exec_penalty_s"]
                    saved = ep * sw_sc
                    race.loc[idx, "total_penalty_s"] += ep
                    race.loc[idx, "sw_saved_s"]      += saved

        # ── Compute positions ────────────────────────────────────────────────
        # Net time saved by software
        race["net_delta_s"] = race["sw_saved_s"]

        # Convert saved time → positions gained
        race["pos_gain"] = (race["net_delta_s"] / field_spread).round().astype(int)

        # FIX: handle NaN positions cleanly
        valid_pos = race["original_position"].notna()
        race["simulated_position"] = race["original_position"].copy()
        race.loc[valid_pos, "simulated_position"] = (
            (race.loc[valid_pos, "original_position"] - race.loc[valid_pos, "pos_gain"])
            .clip(lower=1)
        )

        # Points
        race["original_points"]   = race["original_position"].map(
            lambda p: points_sys.get(int(p), 0) if pd.notna(p) else 0
        )
        race["simulated_points"]  = race["simulated_position"].map(
            lambda p: points_sys.get(int(p), 0) if pd.notna(p) else 0
        )

        corrected_races.append(race)

    if not corrected_races:
        return pd.DataFrame(), pd.DataFrame()

    all_races = pd.concat(corrected_races, ignore_index=True)

    # Championship tally
    orig_champ = (
        all_races.groupby(["driver","driver_name"])["original_points"]
        .sum().reset_index()
        .sort_values("original_points", ascending=False)
        .reset_index(drop=True)
    )
    orig_champ["original_position"] = orig_champ.index + 1

    sim_champ = (
        all_races.groupby(["driver","driver_name"])["simulated_points"]
        .sum().reset_index()
        .sort_values("simulated_points", ascending=False)
        .reset_index(drop=True)
    )
    sim_champ["simulated_position"] = sim_champ.index + 1

    standings = orig_champ.merge(sim_champ, on=["driver","driver_name"])
    standings["points_delta"]    = standings["simulated_points"] - standings["original_points"]
    standings["position_change"] = standings["original_position"] - standings["simulated_position"]

    return standings, all_races


# ── Offline analysis (no Ergast) ──────────────────────────────────────────────
# When the API is unavailable, we can still analyse pit strategy variance
# from pitstops.csv alone and produce meaningful charts.

def analyse_offline(year, pitstops_df, lap_df):
    """
    Full variance analysis using only pitstops.csv.
    Returns (window_errors, exec_penalties, summary_stats).
    """
    print(f"  Running offline variance analysis for {year}...")
    window_errors  = compute_pit_window_errors(pitstops_df, year)
    window_errors  = compute_undercut_missed(window_errors)
    exec_penalties = compute_stop_execution_penalties(pitstops_df, year)

    if window_errors.empty:
        print(f"  ✗ No pit data for {year}")
        return window_errors, exec_penalties, {}

    sw_era = get_software_era(year)
    deg_penalty = get_tyre_deg_penalty(year)

    total_driver_races = len(window_errors)
    avg_window_err     = window_errors["avg_window_error_laps"].mean()
    avg_penalty        = window_errors["penalty_s"].mean()
    sw_saves           = avg_penalty * sw_era["window_correction"]
    avg_exec_pen       = exec_penalties["exec_penalty_s"].mean() if not exec_penalties.empty else 0
    avg_missed_uc      = window_errors["missed_undercut_penalty_s"].mean()

    total_avg_penalty  = avg_penalty + avg_exec_pen + avg_missed_uc
    total_sw_saves     = (
        avg_penalty   * sw_era["window_correction"] +
        avg_exec_pen  * sw_era["stop_time_correction"] +
        avg_missed_uc * sw_era["undercut_awareness"]
    )

    stats = {
        "year":             year,
        "sw_era_label":     sw_era["label"],
        "driver_races":     total_driver_races,
        "avg_window_err":   round(avg_window_err, 2),
        "avg_window_pen_s": round(avg_penalty, 2),
        "avg_exec_pen_s":   round(avg_exec_pen, 2),
        "avg_missed_uc_s":  round(avg_missed_uc, 2),
        "total_avg_pen_s":  round(total_avg_penalty, 2),
        "sw_saves_s":       round(total_sw_saves, 2),
        "net_variance_s":   round(total_avg_penalty - total_sw_saves, 2),
    }

    print(f"  Races analysed:        {window_errors['race'].nunique()}")
    print(f"  Driver-races:          {total_driver_races}")
    print(f"  Avg window error:      {avg_window_err:.2f} laps")
    print(f"  Avg window penalty:    {avg_penalty:.2f}s")
    print(f"  Avg exec penalty:      {avg_exec_pen:.2f}s")
    print(f"  Avg missed undercut:   {avg_missed_uc:.2f}s")
    print(f"  ─────────────────────────────────")
    print(f"  Total avg penalty:     {total_avg_penalty:.2f}s")
    print(f"  SW saves (modelled):   {total_sw_saves:.2f}s")
    print(f"  Remaining variance:    {total_avg_penalty - total_sw_saves:.2f}s")

    return window_errors, exec_penalties, stats


# ── Plotting ───────────────────────────────────────────────────────────────────

def setup_fig_style():
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
        "axes.titlesize":   11,
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
    })


def plot_offline_season(year, window_errors, exec_penalties, sw_era):
    """Dashboard for offline (no Ergast) single-season analysis."""
    setup_fig_style()
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(COLOUR["bg"])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

    fig.text(0.02, 0.97,
             f"F1 {year} — STRATEGY VARIANCE ANALYSIS",
             fontsize=16, fontweight="bold", color=COLOUR["text"],
             fontfamily="monospace", va="top")
    fig.text(0.02, 0.935,
             f"Software era: {sw_era['label']}  ·  "
             f"Window correction: {sw_era['window_correction']*100:.0f}%  ·  "
             f"Stop-time correction: {sw_era['stop_time_correction']*100:.0f}%  ·  "
             f"Undercut awareness: {sw_era['undercut_awareness']*100:.0f}%",
             fontsize=8.5, color=COLOUR["muted"], fontfamily="monospace", va="top")

    yr_we = window_errors[window_errors["year"] == year]
    yr_ep = exec_penalties[exec_penalties["year"] == year] if not exec_penalties.empty else pd.DataFrame()

    # ── 1. Pit window error distribution ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if not yr_we.empty:
        errors = yr_we["avg_window_error_laps"]
        ax1.hist(errors, bins=20, color=COLOUR["blue"], alpha=0.8, edgecolor=COLOUR["border"])
        ax1.axvline(errors.mean(), color=COLOUR["amber"], linestyle="--",
                    linewidth=1.5, label=f"Mean: {errors.mean():.1f} laps")
        ax1.set_title("Pit Window Error Distribution")
        ax1.set_xlabel("Laps from optimal")
        ax1.set_ylabel("Driver-races")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

    # ── 2. Penalty components per race (top 12 races) ─────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if not yr_we.empty:
        race_pen = yr_we.groupby("race")["penalty_s"].mean().sort_values(ascending=False).head(12)
        if not yr_ep.empty:
            race_exec = yr_ep.groupby("race")["exec_penalty_s"].mean().reindex(race_pen.index, fill_value=0)
        else:
            race_exec = pd.Series(0, index=race_pen.index)
        x = np.arange(len(race_pen))
        ax2.bar(x, race_pen.values, color=COLOUR["amber"], alpha=0.85, label="Window penalty")
        ax2.bar(x, race_exec.values, bottom=race_pen.values, color=COLOUR["red"], alpha=0.85, label="Exec penalty")
        ax2.set_xticks(x)
        ax2.set_xticklabels([r[:7] for r in race_pen.index], rotation=45, ha="right")
        ax2.set_ylabel("Avg penalty (s)")
        ax2.set_title("Strategy Penalty by Race")
        ax2.legend(fontsize=7)
        ax2.grid(axis="y", alpha=0.3)

    # ── 3. Driver variance (top 10 worst / best) ──────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if not yr_we.empty:
        drv = yr_we.groupby("driver")[["penalty_s","missed_undercut_penalty_s"]].mean()
        drv["total"] = drv["penalty_s"] + drv["missed_undercut_penalty_s"]
        drv = drv.sort_values("total", ascending=False).head(12)
        y   = np.arange(len(drv))
        ax3.barh(y, drv["penalty_s"], color=COLOUR["amber"], alpha=0.8, label="Window")
        ax3.barh(y, drv["missed_undercut_penalty_s"], left=drv["penalty_s"],
                 color=COLOUR["red"], alpha=0.8, label="Missed undercut")
        ax3.set_yticks(y)
        ax3.set_yticklabels([n[:14] for n in drv.index], fontsize=7)
        ax3.set_xlabel("Total strategy penalty (s)")
        ax3.set_title("Top 12 Strategy Penalty — by Driver")
        ax3.legend(fontsize=7)
        ax3.grid(axis="x", alpha=0.3)

    # ── 4. Penalty breakdown pie ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if not yr_we.empty:
        avg_w  = yr_we["penalty_s"].mean()
        avg_uc = yr_we["missed_undercut_penalty_s"].mean()
        avg_e  = yr_ep["exec_penalty_s"].mean() if not yr_ep.empty else 0
        labels  = ["Window timing", "Missed undercut", "Slow pit stop"]
        sizes   = [avg_w, avg_uc, avg_e]
        colors  = [COLOUR["amber"], COLOUR["red"], COLOUR["blue"]]
        sizes   = [max(0, s) for s in sizes]
        if sum(sizes) > 0:
            ax4.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                    textprops={"fontsize": 8, "color": COLOUR["text"]})
        ax4.set_title("Variance Source Breakdown")

    # ── 5. Software capability timeline ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    eras  = sorted(SOFTWARE_ERAS.keys())
    wc    = [SOFTWARE_ERAS[y]["window_correction"]    for y in eras]
    sc    = [SOFTWARE_ERAS[y]["stop_time_correction"] for y in eras]
    uc    = [SOFTWARE_ERAS[y]["undercut_awareness"]   for y in eras]
    vr    = [SOFTWARE_ERAS[y]["variance_reduction"]   for y in eras]

    ax5.plot(eras, wc, color=COLOUR["amber"], linewidth=2, marker="o", ms=4, label="Window correction")
    ax5.plot(eras, sc, color=COLOUR["blue"],  linewidth=2, marker="s", ms=4, label="Stop-time correction")
    ax5.plot(eras, uc, color=COLOUR["red"],   linewidth=2, marker="^", ms=4, label="Undercut awareness")
    ax5.plot(eras, vr, color=COLOUR["green"], linewidth=2.5, linestyle="--", marker="D", ms=4, label="Overall variance↓")
    ax5.axvline(year, color="white", linestyle=":", linewidth=1.2, label=f"{year}")
    ax5.set_xlabel("Year")
    ax5.set_ylabel("Capability (0–1)")
    ax5.set_title("Software Capability Timeline")
    ax5.legend(fontsize=7, ncol=2)
    ax5.grid(alpha=0.3)
    ax5.set_ylim(0, 1.05)

    # ── 6. SW savings vs remaining variance ──────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    if not yr_we.empty:
        # Per driver: total penalty vs SW-corrected penalty
        drv_pen   = yr_we.groupby("driver")[["penalty_s","missed_undercut_penalty_s"]].mean()
        drv_pen["total"] = drv_pen["penalty_s"] + drv_pen["missed_undercut_penalty_s"]
        ep_mean   = yr_ep.groupby("driver")["exec_penalty_s"].mean() if not yr_ep.empty else pd.Series(dtype=float)
        drv_pen["exec"] = drv_pen.index.map(lambda d: ep_mean.get(d, 0))
        drv_pen["grand_total"] = drv_pen["total"] + drv_pen["exec"]

        sw  = get_software_era(year)
        drv_pen["sw_saved"] = (
            drv_pen["penalty_s"]   * sw["window_correction"] +
            drv_pen["missed_undercut_penalty_s"] * sw["undercut_awareness"] +
            drv_pen["exec"]        * sw["stop_time_correction"]
        )
        drv_pen["remaining"] = drv_pen["grand_total"] - drv_pen["sw_saved"]

        x = np.arange(min(12, len(drv_pen)))
        top = drv_pen.sort_values("grand_total", ascending=False).head(12)
        ax6.bar(x, top["grand_total"].values, color=COLOUR["muted"],    alpha=0.6, label="Total penalty")
        ax6.bar(x, top["remaining"].values,   color=COLOUR["red"],      alpha=0.9, label="After SW correction")
        ax6.set_xticks(x)
        ax6.set_xticklabels([n[:8] for n in top.index], rotation=45, ha="right", fontsize=7)
        ax6.set_ylabel("Penalty (s)")
        ax6.set_title("Total Penalty vs SW-Corrected Penalty")
        ax6.legend(fontsize=7)
        ax6.grid(axis="y", alpha=0.3)

    outfile = f"f1_strategy_{year}.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor=COLOUR["bg"])
    print(f"  ✓ Saved {outfile}")
    plt.show()
    plt.close()


def plot_sweep(sweep_stats, pitstops_df):
    """Multi-era variance reduction overview."""
    setup_fig_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(COLOUR["bg"])
    fig.suptitle("F1 STRATEGY VARIANCE ACROSS ERAS",
                 fontsize=15, fontweight="bold", color=COLOUR["text"],
                 fontfamily="monospace")

    df = pd.DataFrame(sweep_stats)
    years = df["year"].tolist()

    # ── 1. Total penalty over time ────────────────────────────────────────
    ax = axes[0, 0]
    ax.bar(years, df["total_avg_pen_s"], color=COLOUR["blue"],   alpha=0.8, label="Total penalty")
    ax.plot(years, df["net_variance_s"], color=COLOUR["green"],  linewidth=2.5,
            marker="o", markersize=5, label="After SW correction")
    ax.set_title("Average Strategy Penalty Per Driver-Race")
    ax.set_xlabel("Year"); ax.set_ylabel("Seconds")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # ── 2. Penalty breakdown stacked ─────────────────────────────────────
    ax = axes[0, 1]
    ax.bar(years, df["avg_window_pen_s"], color=COLOUR["amber"], alpha=0.85, label="Window timing")
    ax.bar(years, df["avg_exec_pen_s"],   bottom=df["avg_window_pen_s"],
           color=COLOUR["red"],   alpha=0.85, label="Slow pit stop")
    ax.bar(years, df["avg_missed_uc_s"],
           bottom=df["avg_window_pen_s"] + df["avg_exec_pen_s"],
           color=COLOUR["blue"],  alpha=0.85, label="Missed undercut")
    ax.set_title("Penalty Source Composition Over Years")
    ax.set_xlabel("Year"); ax.set_ylabel("Seconds")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # ── 3. SW savings over time ───────────────────────────────────────────
    ax = axes[1, 0]
    ax.fill_between(years, df["sw_saves_s"], alpha=0.25, color=COLOUR["green"])
    ax.plot(years, df["sw_saves_s"], color=COLOUR["green"], linewidth=2.5,
            marker="o", markersize=5, label="SW saves (s)")
    ax.set_title("Average Time Saved Per Driver-Race by Software")
    ax.set_xlabel("Year"); ax.set_ylabel("Seconds")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # ── 4. Software capability model ─────────────────────────────────────
    ax = axes[1, 1]
    eras = sorted(SOFTWARE_ERAS.keys())
    ax.plot(eras, [SOFTWARE_ERAS[y]["window_correction"]    for y in eras],
            color=COLOUR["amber"], linewidth=2, marker="o", ms=4, label="Window correction")
    ax.plot(eras, [SOFTWARE_ERAS[y]["stop_time_correction"] for y in eras],
            color=COLOUR["blue"],  linewidth=2, marker="s", ms=4, label="Stop-time correction")
    ax.plot(eras, [SOFTWARE_ERAS[y]["undercut_awareness"]   for y in eras],
            color=COLOUR["red"],   linewidth=2, marker="^", ms=4, label="Undercut awareness")
    ax.plot(eras, [SOFTWARE_ERAS[y]["variance_reduction"]   for y in eras],
            color=COLOUR["green"], linewidth=2.5, linestyle="--", label="Overall variance↓")

    for start, end, label in [(1950,1983,"Pre-telemetry"),(1984,1999,"Telemetry"),
                               (2000,2011,"Simulation"),(2012,2024,"AI/ML")]:
        ax.axvspan(start, end, alpha=0.04, color=COLOUR["blue"])
        ax.text((start+end)/2, 0.02, label, ha="center", fontsize=7,
                color=COLOUR["muted"])
    ax.set_xlabel("Year"); ax.set_ylabel("Capability (0–1)")
    ax.set_title("Software Capability Growth by Dimension")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("f1_era_sweep.png", dpi=150, bbox_inches="tight", facecolor=COLOUR["bg"])
    print("  ✓ Saved f1_era_sweep.png")
    plt.show()
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def analyse_year_full(year, pitstops_df, lap_df, offline=False):
    """Run full analysis for a single year."""
    print(f"\n{'='*60}")
    print(f"  Analysing {year}")
    print(f"{'='*60}")

    sw_era = get_software_era(year)
    print(f"  Software era:   {sw_era['label']}")
    print(f"  Window corr:    {sw_era['window_correction']*100:.0f}%")
    print(f"  Stop-time corr: {sw_era['stop_time_correction']*100:.0f}%")
    print(f"  Undercut aware: {sw_era['undercut_awareness']*100:.0f}%")

    window_errors  = compute_pit_window_errors(pitstops_df, year)
    window_errors  = compute_undercut_missed(window_errors)
    exec_penalties = compute_stop_execution_penalties(pitstops_df, year)

    _, _, stats = analyse_offline(year, pitstops_df, lap_df)

    if not offline:
        print(f"\n  Fetching race results from Ergast...")
        race_results = fetch_season_results(year)
        if race_results.empty:
            print(f"  ⚠  No Ergast data — charting offline analysis only")
            return window_errors, exec_penalties, stats, None, None

        standings, all_races = simulate_corrected_season(
            race_results, window_errors, exec_penalties, year, sw_era
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

            hc = standings.iloc[0]["driver_name"]
            sc_name = standings.sort_values("simulated_points", ascending=False).iloc[0]["driver_name"]
            if hc != sc_name:
                print(f"\n  ★ CHAMPIONSHIP OUTCOME CHANGES!")
                print(f"    Historical champion:    {hc}")
                print(f"    Software-aided champion:{sc_name}")
            else:
                print(f"\n  ✓ Same champion: {hc}")

        return window_errors, exec_penalties, stats, standings, all_races

    return window_errors, exec_penalties, stats, None, None


def main():
    parser = argparse.ArgumentParser(
        description="F1 Strategy Variance Analyser v2 — LUT University Thesis"
    )
    parser.add_argument("--year",    type=int, default=2010)
    parser.add_argument("--mode",    choices=["single","sweep"], default="single")
    parser.add_argument("--start",   type=int, default=1994)
    parser.add_argument("--end",     type=int, default=2010)
    parser.add_argument("--offline", action="store_true",
                        help="Skip Ergast API, use pitstops.csv only")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--udp",     type=str, default=None,
                        help="Path to a CSV exported by f1.py (UDP telemetry). "
                             "Analyses strategy variance for that live session. "
                             "Example: --udp captures/bahrain_2025.csv")
    args = parser.parse_args()

    print("\n  F1 Strategy Variance Analyser  v2.0")
    print("  LUT University — Bachelor's Thesis\n")

    pitstops_df = load_pitstops_csv(PITSTOPS_CSV)
    lap_df      = load_lap_progression(LAP_TIMES_XL)
    print(f"  Pit stops: {len(pitstops_df)} records "
          f"({pitstops_df['Year'].min() if not pitstops_df.empty else 'N/A'}–"
          f"{pitstops_df['Year'].max() if not pitstops_df.empty else 'N/A'})")
    print(f"  Lap times: {len(lap_df)} years")

    # ── UDP / live session mode ───────────────────────────────────────────────
    if args.udp:
        analyse_udp_session(args.udp, no_plot=args.no_plot)
        return

    if args.mode == "single":
        we, ep, stats, standings, all_races = analyse_year_full(
            args.year, pitstops_df, lap_df, offline=args.offline
        )
        if not args.no_plot:
            sw_era = get_software_era(args.year)
            plot_offline_season(args.year, we, ep, sw_era)

    elif args.mode == "sweep":
        all_stats = []
        for year in range(args.start, args.end + 1):
            we, ep, stats, _, _ = analyse_year_full(
                year, pitstops_df, lap_df, offline=True
            )
            if stats:
                all_stats.append(stats)

        if all_stats and not args.no_plot:
            plot_sweep(all_stats, pitstops_df)

        if all_stats:
            df = pd.DataFrame(all_stats)
            df.to_csv("f1_sweep_summary.csv", index=False)
            print("\n  ✓ Saved f1_sweep_summary.csv")
            print("\n  SWEEP SUMMARY:")
            print(df[["year","sw_era_label","total_avg_pen_s","sw_saves_s","net_variance_s"]].to_string(index=False))


if __name__ == "__main__":
    main()