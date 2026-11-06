import numpy as np
import pandas as pd


def weekly_temperature(temps):
    calendar = temps["date"].dt.isocalendar()
    frame = temps.assign(iso_year=calendar["year"].astype(int), iso_week=calendar["week"].astype(int))
    grouped = frame.groupby(["region", "iso_year", "iso_week"])["tmax"]
    return pd.DataFrame({"tmax_mean": grouped.mean(), "tmax_peak": grouped.max()}).reset_index()


def expected_deaths(deaths, reference_years, window=1, min_years=3):
    frame = deaths.sort_values(["region", "iso_year", "iso_week"]).copy()
    frame["smoothed"] = frame.groupby(["region", "iso_year"])["deaths"].transform(
        lambda series: series.rolling(2 * window + 1, center=True, min_periods=1).mean()
    )
    pivot = frame.pivot_table(index=["region", "iso_week"], columns="iso_year", values="smoothed")
    reference = [year for year in reference_years if year in pivot.columns]
    expected = []
    for year in sorted(frame["iso_year"].unique()):
        usable = [y for y in reference if y != year]
        block = pivot[usable] if usable else pd.DataFrame(index=pivot.index)
        counts = block.notna().sum(axis=1)
        mean = block.mean(axis=1).where(counts >= min_years)
        expected.append(pd.DataFrame({"region": mean.index.get_level_values(0), "iso_week": mean.index.get_level_values(1), "iso_year": year, "expected": mean.values}))
    expected = pd.concat(expected, ignore_index=True)
    merged = frame.drop(columns="smoothed").merge(expected, on=["region", "iso_year", "iso_week"], how="left")
    merged["excess"] = merged["deaths"] / merged["expected"] - 1.0
    return merged


def build_panel(raw, config):
    deaths = expected_deaths(raw["deaths"], config["reference_years"], config.get("baseline_window", 1), config.get("baseline_min_years", 3))
    temps = weekly_temperature(raw["temps"])
    panel = deaths.merge(temps, on=["region", "iso_year", "iso_week"], how="inner")
    train_years = set(config["train_years"])
    climatology = panel[panel["iso_year"].isin(train_years)].groupby(["region", "iso_week"])["tmax_mean"].mean().rename("tmax_clim")
    panel = panel.merge(climatology.reset_index(), on=["region", "iso_week"], how="left")
    panel["temp_anomaly"] = panel["tmax_mean"] - panel["tmax_clim"]
    panel = panel.sort_values(["region", "iso_year", "iso_week"]).reset_index(drop=True)
    lag = int(config.get("excess_reporting_lag", 2))
    by_region = panel.groupby("region")
    panel["excess_lagged"] = by_region["excess"].shift(lag)
    panel["temp_anomaly_previous"] = by_region["temp_anomaly"].shift(1)
    panel["event"] = (panel["excess"] >= config["event_threshold"]).astype(int)
    weeks = config["season_weeks"]
    panel["in_season"] = panel["iso_week"].between(weeks[0], weeks[1])
    covariates = raw["covariates"].set_index("region")
    for column in covariates.columns:
        panel[column] = panel["region"].map(covariates[column])
    panel["country"] = panel["region"].map(raw["regions"].set_index("region")["country"])
    return panel


def split(panel, config):
    usable = panel[panel["in_season"] & panel["expected"].notna() & panel["excess_lagged"].notna() & panel["temp_anomaly_previous"].notna()]
    parts = {}
    for name in ("train", "validation", "test"):
        years = config[f"{name}_years"]
        parts[name] = usable[usable["iso_year"].isin(years)].reset_index(drop=True)
    return parts


def coverage(panel, parts):
    rows = []
    for name, frame in parts.items():
        rows.append(
            {
                "split": name,
                "years": ", ".join(str(y) for y in sorted(frame["iso_year"].unique())),
                "regions": int(frame["region"].nunique()),
                "region_weeks": int(len(frame)),
                "events": int(frame["event"].sum()),
                "event_rate": float(frame["event"].mean()) if len(frame) else np.nan,
            }
        )
    return pd.DataFrame(rows)
