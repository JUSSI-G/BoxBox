"""
F1 Strategy Variance Analyser

Loads pit stop data, computes window/execution penalties, and drives xgb.py.

Usage:
    python analyser.py
    python analyser.py --start 1994 --end 2024
    python analyser.py --no-plot
"""

import pandas as pd
import numpy as np
import ast, argparse, os, json, time, sys
import requests

_HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(_HERE, "data")
OUTPUTS_DIR = os.path.join(_HERE, "outputs")
CACHE_DIR   = os.path.join(DATA_DIR, "f1_cache")

os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,   exist_ok=True)

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ERGAST_BASE  = "https://api.jolpi.ca/ergast/f1"
PITSTOPS_CSV = os.path.join(DATA_DIR, "pitstops.csv")
KAGGLE_CSV   = os.path.join(DATA_DIR, "Formula1_Pitstop_Data_1950-2024_all_rounds.csv")

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

def position_to_points(position, year):
    ps = get_points_system(year)
    try:
        return ps.get(int(position), 0)
    except Exception:
        return 0

# ── Tyre degradation constants ────────────────────────────────────────────
TYRE_DEG_PENALTY_S_PER_LAP = {
    1994: 0.45,
    2000: 0.42,
    2005: 0.38,
    2007: 0.35,
    2010: 0.30,
    2011: 0.28,
    2014: 0.25,
    2017: 0.22,
    2022: 0.20,
}

def get_tyre_deg_penalty(year):
    applicable = {y: v for y, v in TYRE_DEG_PENALTY_S_PER_LAP.items() if y <= year}
    return applicable[max(applicable.keys())]

TYPICAL_RACE_LAPS = {
    1994:62, 1995:62, 1996:62, 1997:62, 1998:62, 1999:62,
    2000:62, 2001:62, 2002:62, 2003:62, 2004:62, 2005:62,
    2006:62, 2007:62, 2008:62, 2009:60, 2010:60,
    2011:58, 2012:58, 2013:60, 2014:58, 2015:58, 2016:58,
    2017:57, 2018:57, 2019:58, 2020:58, 2021:57, 2022:57,
    2023:57, 2024:57,
}

ERA_LABELS = {
    1950: "Pre-digital",
    1975: "Early electronics",
    1984: "Telemetry begins",
    1994: "Real-time data",
    2000: "Simulation tools",
    2006: "Full analytics",
    2012: "Predictive models",
    2018: "AI / Monte Carlo",
}

def get_era_label(year):
    applicable = {y: v for y, v in ERA_LABELS.items() if y <= year}
    return applicable[max(applicable.keys())]


# ── Historical wet race lookup (1994–2017) ─────────────────────────────────────
# 2018+ is covered by FastF1 in fetch_race_conditions()

WET_RACE_ROUNDS = {
    (1994, 15),   # Japan
    (1995,  3), (1995,  7), (1995, 11), (1995, 14), (1995, 16),
    (1996,  2), (1996,  6), (1996,  7),
    (1997,  5), (1997,  8), (1997, 12),
    (1998,  3), (1998,  9), (1998, 13),
    (1999,  7), (1999, 14),
    (2000,  6), (2000,  8), (2000, 11), (2000, 13), (2000, 15), (2000, 16),
    (2001,  2), (2001,  3),
    (2002, 10),
    (2003,  1), (2003,  3),
    (2004,  2), (2004, 15), (2004, 18),
    (2005, 16),
    (2006, 13), (2006, 16),
    (2007, 10), (2007, 15), (2007, 16),
    (2008,  6), (2008,  8), (2008,  9), (2008, 13), (2008, 14), (2008, 18),
    (2009,  2), (2009,  3),
    (2010,  2), (2010,  4), (2010,  7), (2010, 13), (2010, 17),
    (2011,  7), (2011,  9), (2011, 10), (2011, 11), (2011, 16),
    (2012,  2), (2012, 20),
    (2013,  2),
    (2014, 11), (2014, 15),
    (2015,  9), (2015, 16),
    (2016,  6), (2016, 10), (2016, 20),
    (2017,  2), (2017, 14),
    # 2018–2024: covered by FastF1 in fetch_race_conditions()
}

# ── FastF1 race conditions (weather + safety car) ─────────────────────────────
FASTF1_MIN_YEAR = 2018

def fetch_race_conditions(year):
    """Fetch wet/dry and SC laps per race via FastF1. Cached to f1_cache/."""
    cache_path = os.path.join(CACHE_DIR, f"conditions_{year}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    if year < FASTF1_MIN_YEAR:
        with open(cache_path, "w") as f:
            json.dump({}, f)
        return {}
    try:
        import fastf1
        fastf1.Cache.enable_cache(CACHE_DIR)
    except ImportError:
        print("  ⚠  fastf1 not installed — pip install fastf1")
        return {}
    conditions = {}
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as exc:
        print(f"  ⚠  FastF1 schedule unavailable for {year} ({exc})")
        return {}
    race_events = schedule[schedule["EventFormat"] != "testing"]
    total = len(race_events)
    for i, (_, event) in enumerate(race_events.iterrows(), 1):
        circuit_key = str(event.get("Location", event.get("EventName", f"race_{i}"))).lower().replace(" ", "_")
        print(f"  → FastF1 {year} [{i}/{total}] {circuit_key}", flush=True)
        try:
            session = fastf1.get_session(year, event["RoundNumber"], "R")
            session.load(weather=True, laps=True, telemetry=False, messages=False)
            wet = 0
            if session.weather_data is not None and not session.weather_data.empty:
                rain_col = "Rainfall" if "Rainfall" in session.weather_data.columns else None
                if rain_col:
                    wet = int(session.weather_data[rain_col].sum() > 0)
            sc_laps = 0
            if session.laps is not None and not session.laps.empty:
                if "TrackStatus" in session.laps.columns:
                    sc_laps = int(session.laps["TrackStatus"].astype(str).str.contains("4|6", na=False).sum())
            conditions[circuit_key] = {"wet_race": wet, "sc_laps": sc_laps}
        except Exception as exc:
            print(f"    ⚠  Could not load session ({exc}) — using defaults")
            conditions[circuit_key] = {"wet_race": 0, "sc_laps": 0}
    with open(cache_path, "w") as f:
        json.dump(conditions, f)
    print(f"  Conditions cached for {year}: {len(conditions)} races")
    return conditions


def load_all_race_conditions(start, end):
    """Load race conditions for all years. Returns {year: {circuit: {wet,sc}}}"""
    all_conditions = {}
    for year in range(start, end + 1):
        if year >= FASTF1_MIN_YEAR:
            print(f"  Fetching race conditions {year} from FastF1...")
        all_conditions[year] = fetch_race_conditions(year)
    fetched = sum(1 for y in range(start, end + 1) if y >= FASTF1_MIN_YEAR)
    print(f"  Race conditions loaded: {fetched} seasons via FastF1 "
          f"({start}–{FASTF1_MIN_YEAR - 1} use historical lookup table)")
    return all_conditions

def fetch_grid_positions(year):
    """
    Fetch starting grid positions from Ergast for one season.
    Returns dict: {(round, driver_lastname): grid_position}
    Cached to f1_cache/grid_{year}.json after first fetch.
    """
    cache_path = os.path.join(CACHE_DIR, f"grid_{year}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    url     = f"{ERGAST_BASE}/{year}/results.json"
    all_races = []
    offset, limit, total = 0, 100, None

    while True:
        try:
            print(f"  → fetching {year} grid positions (offset={offset})")
            time.sleep(1.5)
            r = requests.get(url, params={"limit": limit, "offset": offset}, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"  ⚠  Ergast unavailable ({type(exc).__name__}) — grid positions skipped for {year}")
            return {}

        races = data["MRData"]["RaceTable"]["Races"]
        all_races.extend(races)
        if total is None:
            total = int(data["MRData"].get("total", len(all_races)))
        offset += limit
        if offset >= total:
            break

    grid_map = {}
    for race in all_races:
        round_num = int(race["round"])
        for r in race.get("Results", []):
            last_name = r["Driver"]["familyName"].lower()
            grid      = int(r.get("grid", 0))
            grid_map[f"{round_num}_{last_name}"] = grid

    with open(cache_path, "w") as f:
        json.dump(grid_map, f)
    print(f"  ✓ {year} grid positions cached ({len(grid_map)} entries)")
    return grid_map


def load_all_grid_positions(start, end):
    """Load grid positions for all years in range. Returns combined dict."""
    all_grids = {}
    for year in range(start, end + 1):
        grids = fetch_grid_positions(year)
        for key, val in grids.items():
            all_grids[f"{year}_{key}"] = val
    return all_grids

# ── Data loaders ───────────────────────────────────────────────────────────────

def load_pitstops_csv(path=PITSTOPS_CSV):
    """Load historical pit stop data 1994-2010."""
    if not os.path.exists(path):
        print(f"  ⚠  {path} not found")
        return pd.DataFrame()
    df = pd.read_csv(path)
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
    df = df[df["pit_s"].between(10, 120)]
    df = df[df["Lap"] > 3]
    print(f"  ✓ pitstops.csv: {len(df)} records "
          f"({df['Year'].min()}–{df['Year'].max()})")
    return df


def load_kaggle_csv(path=KAGGLE_CSV):
    """
    Load the Kaggle F1 dataset.
    Returns two DataFrames:
      - race_results: position + points per driver per race (all years)
      - pitstops_new: pit stop timing data for 2011+ (parsed from PitStops column)
    """
    if not os.path.exists(path):
        print(f"  ⚠  {path} not found")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(path)

    # ── Race results (positions for all years) ───────────────────────────────
    results_rows = []
    for _, row in df.iterrows():
        year = int(row["Season"])
        points = position_to_points(row["Position"], year)
        results_rows.append({
            "year":        year,
            "round":       int(row["Round"]),
            "circuit":     str(row["Circuit"]),
            "driver_name": str(row["Driver"]),
            "constructor": str(row["Constructor"]),
            "position":    int(row["Position"]) if pd.notna(row["Position"]) else None,
            "laps":        int(row["Laps"]) if pd.notna(row["Laps"]) else 0,
            "points":      points,
        })
    race_results = pd.DataFrame(results_rows)

    # ── Pit stop data for 2011+ (parse PitStops JSON column) ─────────────────
    pit_rows = []
    post2010 = df[df["Season"] >= 2011].copy()
    for _, row in post2010.iterrows():
        try:
            stops = ast.literal_eval(str(row["PitStops"]))
        except Exception:
            continue
        if not stops:
            continue
        year    = int(row["Season"])
        circuit = str(row["Circuit"])
        driver  = str(row["Driver"])
        for stop in stops:
            lap  = stop.get("Lap")
            time = stop.get("StopTime")
            if lap is None or time is None:
                continue
            pit_rows.append({
                "Year":       year,
                "Race":       circuit,
                "DriverName": driver,
                "Lap":        int(lap),
                "Time":       float(time),
                "pit_s":      float(time),
            })

    pitstops_new = pd.DataFrame(pit_rows)
    if not pitstops_new.empty:
        pitstops_new = pitstops_new[pitstops_new["pit_s"].between(10, 120)]
        pitstops_new = pitstops_new[pitstops_new["Lap"] > 3]

    print(f"  ✓ Kaggle CSV: {len(race_results)} race result rows "
          f"({race_results['year'].min()}–{race_results['year'].max()})")
    print(f"  ✓ Kaggle CSV: {len(pitstops_new)} pit stop records 2011+")
    return race_results, pitstops_new


def combine_pitstops(old_df, new_df):
    """Merge pitstops.csv (1994-2010) with Kaggle pit data (2011+)."""
    if old_df.empty and new_df.empty:
        return pd.DataFrame()
    if old_df.empty:
        return new_df
    if new_df.empty:
        return old_df
    combined = pd.concat([old_df, new_df], ignore_index=True)
    print(f"  ✓ Combined pit stops: {len(combined)} records "
          f"({combined['Year'].min()}–{combined['Year'].max()})")
    return combined

# ── Variance calculations ──────────────────────────────────────────────────────

def tyre_deg_loss(lap, total_laps, deg_rate):
    return deg_rate * (lap ** 2) / total_laps


def optimal_pit_laps(n_stops, total_laps, deg_rate):
    if n_stops == 0:
        return []
    if n_stops > 4:
        return [int(total_laps * i / (n_stops + 1)) for i in range(1, n_stops + 1)]
    best_cost, best_laps = float("inf"), None
    def search(stops_left, start_lap, chosen):
        nonlocal best_cost, best_laps
        if stops_left == 0:
            pit_laps   = chosen + [total_laps]
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
    search(n_stops, 3, [])
    return best_laps or [int(total_laps * i / (n_stops + 1))
                         for i in range(1, n_stops + 1)]


def compute_pit_window_errors(pitstops_df, year, race_laps=None):
    if pitstops_df.empty or "Year" not in pitstops_df.columns:
        return pd.DataFrame()
    year_df    = pitstops_df[pitstops_df["Year"] == year].copy()
    if year_df.empty:
        return pd.DataFrame()
    deg_rate   = get_tyre_deg_penalty(year)
    results    = []
    for (race, driver), grp in year_df.groupby(["Race", "DriverName"]):
        grp         = grp.sort_values("Lap")
        actual_laps = grp["Lap"].tolist()
        n_stops     = len(actual_laps)
        race_key    = race.lower().strip()
        total_laps  = (race_laps or {}).get(race_key) or TYPICAL_RACE_LAPS.get(year, 60)
        opt         = optimal_pit_laps(n_stops, total_laps, deg_rate)
        errors      = [abs(a - o) for a, o in zip(actual_laps, opt)]
        avg_err     = float(np.mean(errors))
        results.append({
            "year":                  year,
            "race":                  race,
            "driver":                driver,
            "n_stops":               n_stops,
            "avg_window_error_laps": round(avg_err, 2),
            "window_penalty_s":      round(avg_err * deg_rate, 2),
        })
    return pd.DataFrame(results)


def compute_exec_penalties(pitstops_df, year):
    if pitstops_df.empty or "Year" not in pitstops_df.columns:
        return pd.DataFrame()
    year_df = pitstops_df[pitstops_df["Year"] == year].copy()
    if year_df.empty:
        return pd.DataFrame()
    results = []
    for race, grp in year_df.groupby("Race"):
        fastest = grp["pit_s"].min()
        for driver, dgrp in grp.groupby("DriverName"):
            results.append({
                "year":           year,
                "race":           race,
                "driver":         driver,
                "exec_penalty_s": round(max(0.0, dgrp["pit_s"].mean() - fastest), 3),
            })
    return pd.DataFrame(results)


def compute_undercut_missed(window_errors):
    if window_errors.empty:
        return window_errors
    we = window_errors.copy()
    we["missed_undercut_s"] = 0.0
    for _, grp in we.groupby("race"):
        near = grp[grp["avg_window_error_laps"] <= 2]
        for idx, row in grp.iterrows():
            if row["avg_window_error_laps"] > 2 and len(near) > 0:
                we.loc[idx, "missed_undercut_s"] = 0.5 * row["n_stops"]
    return we

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="F1 Strategy Variance Analyser"
    )
    parser.add_argument("--start",   type=int, default=1994)
    parser.add_argument("--end",     type=int, default=2024)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("\n  F1 Strategy Variance Analyser\n")

    # Load data
    pitstops_old          = load_pitstops_csv(PITSTOPS_CSV)
    race_results, pitstops_new = load_kaggle_csv(KAGGLE_CSV)
    pitstops_df           = combine_pitstops(pitstops_old, pitstops_new)

    if pitstops_df.empty:
        print("  ✗ No pit stop data — cannot continue")
        return
    if race_results.empty:
        print("  ✗ No race results — cannot continue")
        return

    # Filter to requested years
    race_results = race_results[
        race_results["year"].between(args.start, args.end)
    ]
    ergast_by_year = {
        year: grp for year, grp in race_results.groupby("year")
    }

    try:
        from xgb import run as run_xgb
    except ImportError:
        print("  ✗ xgb.py not found — make sure it is in the same folder as analyser.py")
        return

    print(f"\n  Fetching grid positions {args.start}–{args.end} from Ergast...")
    grid_positions = load_all_grid_positions(args.start, args.end)
    print(f"  ✓ Grid positions loaded: {len(grid_positions)} entries")
    print(f"  Race conditions loading (FastF1 {FASTF1_MIN_YEAR}–{args.end} + historical table)...")
    race_conditions = load_all_race_conditions(args.start, args.end)

    run_xgb(
        pitstops_df=pitstops_df,
        ergast_results_by_year=ergast_by_year,
        grid_positions=grid_positions,
        race_conditions=race_conditions,
        start=args.start,
        end=args.end,
        outputs_dir=OUTPUTS_DIR,
    )


if __name__ == "__main__":
    main()