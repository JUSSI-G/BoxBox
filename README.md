# BoxBox

F1 strategy analysis tool. Computes pit stop variance metrics from historical data, trains an XGBoost model to predict race points, and simulates what-if championships under perfect pit strategy.

## What it does

1. **Variance analysis**: loads pit stop data (1994–2024), computes pit window errors and execution penalties per driver per race
2. **XGBoost model**: trains on 1994–2011, tests on 2012–2024, runs per-era feature importance analysis
3. **Championship simulation**: zeros variance features (keeps car quality fixed), checks which seasons the title would have changed under perfect strategy
4. **Web dashboard**: interactive UI at `http://localhost:5001` with charts and live pipeline output

## Data

Place these files in `data/`:

| File | Description |
|------|-------------|
| `pitstops.csv` | Pit stop timing 1994–2010 |
| `Formula1_Pitstop_Data_1950-2024_all_rounds.csv` | Positions + pit data 2011–2024 |

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/JUSSI-G/BoxBox.git
cd BoxBox
```

**2. Create and activate a virtual environment**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**3. Install dependencies**
```powershell
pip install -r requirements.txt
```

## Usage

### Web dashboard (recommended)

```powershell
python server.py
```

Open [http://localhost:5001](http://localhost:5001). Use the sidebar to run the full pipeline or simulate a single season.

### Command line

Run the full pipeline:
```powershell
python analyser.py --start 1994 --end 2024
```

Simulate a single season (requires existing dataset):
```powershell
python f1.py --year 2010 --dataset outputs/f1_xgb_dataset.csv
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--start` | 1994 | First season to analyse |
| `--end` | 2024 | Last season to analyse |
| `--no-plot` | — | Skip interactive plots |
| `--year` | — | Single season (f1.py only) |

## Pipeline

```
analyser.py  →  xgb.py  →  f1.py
```

| File | Role | Outputs |
|------|------|---------|
| `analyser.py` | Loads CSVs, computes window/execution penalties | feeds xgb.py |
| `xgb.py` | Trains XGBoost, era importance analysis | `f1_xgb_dataset.csv`, `f1_era_importance.json` |
| `f1.py` | Championship counterfactual simulation | `f1_simulation_sweep.csv` |
| `server.py` | Flask web server + SSE streaming | serves dashboard |

## Requirements

See `requirements.txt`. Key dependencies: `flask`, `xgboost`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `fastf1`, `requests`.
