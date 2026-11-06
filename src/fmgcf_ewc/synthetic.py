import numpy as np
import pandas as pd


def generate(countries=("AA", "BB", "CC"), regions_per_country=8, start_year=2015, end_year=2025, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for c_index, country in enumerate(countries):
        centre_lat = 37.0 + 3.0 * c_index
        centre_lon = 15.0 + 6.0 * c_index
        for r in range(regions_per_country):
            rows.append(
                {
                    "region": f"{country}{r:02d}",
                    "name": f"Region {country}-{r}",
                    "country": country,
                    "lat": centre_lat + rng.normal(0, 1.2),
                    "lon": centre_lon + rng.normal(0, 1.5),
                }
            )
    regions = pd.DataFrame(rows)
    n = len(regions)
    share65 = np.clip(rng.normal(21.0, 3.0, n), 14, 30)
    gdp = np.clip(rng.normal(25000, 6000, n), 12000, 45000)
    covariates = pd.DataFrame({"region": regions["region"], "share65": share65, "gdp_pps": gdp})
    sensitivity = 0.010 + 0.0025 * (share65 - 21.0) - 0.0000012 * (gdp - 25000)
    sensitivity = np.clip(sensitivity, 0.002, None)
    population_scale = rng.uniform(250, 1500, n)
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    day = dates.day_of_year.values
    temps = []
    heat = np.zeros((n, len(dates)))
    for i in range(n):
        base = 22.0 - 0.6 * (regions.loc[i, "lat"] - 37.0) + 10.0 * np.sin(2 * np.pi * (day - 110) / 365.25)
        noise = rng.normal(0, 2.0, len(dates))
        tmax = base + noise
        temps.append(pd.DataFrame({"region": regions.loc[i, "region"], "date": dates, "tmax": tmax}))
        heat[i] = tmax
    for year in range(start_year, end_year + 1):
        waves = rng.integers(1, 4)
        for _ in range(waves):
            centre = rng.integers(170, 235)
            length = rng.integers(5, 14)
            amplitude = rng.uniform(3.0, 8.0)
            mask = (dates.year == year) & (day >= centre) & (day < centre + length)
            affected = rng.random(n) < 0.7
            heat[np.ix_(affected, mask)] += amplitude
    for i in range(n):
        temps[i]["tmax"] = heat[i]
    temps = pd.concat(temps, ignore_index=True)
    weekly = temps.assign(iso_year=temps["date"].dt.isocalendar().year.astype(int), iso_week=temps["date"].dt.isocalendar().week.astype(int))
    weekly = weekly.groupby(["region", "iso_year", "iso_week"], as_index=False)["tmax"].mean()
    climatology = weekly.groupby(["region", "iso_week"])["tmax"].transform("mean")
    anomaly = (weekly["tmax"] - climatology).clip(lower=0)
    index = weekly["region"].map({code: k for k, code in enumerate(regions["region"])})
    seasonal = 1.0 + 0.18 * np.cos(2 * np.pi * (weekly["iso_week"] - 2) / 52.0)
    covid = weekly["iso_year"].isin([2020, 2021, 2022]) * 0.15
    trend = 0.005 * (weekly["iso_year"] - start_year)
    expected = population_scale[index] * seasonal * (1 + covid + trend)
    effect = 1 + sensitivity[index] * anomaly.values * 4.0
    deaths = rng.poisson(expected * effect)
    deaths_frame = weekly[["region", "iso_year", "iso_week"]].assign(deaths=deaths.astype(float))
    return {
        "regions": regions,
        "deaths": deaths_frame,
        "temps": temps[["region", "date", "tmax"]],
        "covariates": covariates,
        "log": [{"source": "synthetic", "regions": n}],
    }
