import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .jsonstat import to_frame

EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
NUTS_POINTS = "https://raw.githubusercontent.com/eurostat/Nuts2json/master/pub/v2/2021/4326/nutspt_{level}.json"
USER_AGENT = "fmgcf-ewc research software (https://github.com/uomoathens/fmgcf-ewc)"


class SourceError(RuntimeError):
    pass


def _cache_path(cache, key):
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in key)
    return Path(cache) / f"{safe[:180]}.json"


def fetch_json(url, cache, key, retries=4, pause=1.0):
    path = _cache_path(cache, key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            time.sleep(pause)
            return payload
        except urllib.error.HTTPError as error:
            last = error
            if error.code in (400, 404):
                raise SourceError(f"{error.code} for {url}") from error
            time.sleep(pause * (2 ** attempt) * 5)
        except (urllib.error.URLError, TimeoutError) as error:
            last = error
            time.sleep(pause * (2 ** attempt) * 5)
    raise SourceError(f"failed after {retries} attempts: {url}: {last}")


def eurostat(code, filters, cache):
    query = [("format", "JSON"), ("lang", "EN")]
    for name, value in filters.items():
        for item in value if isinstance(value, (list, tuple)) else [value]:
            query.append((name, str(item)))
    url = EUROSTAT + code + "?" + urllib.parse.urlencode(query)
    key = "eurostat_" + code + "_" + "_".join(f"{k}-{v}" for k, v in query[2:])
    return to_frame(fetch_json(url, cache, key))


def nuts_points(level, countries, cache):
    payload = fetch_json(NUTS_POINTS.format(level=level), cache, f"nuts_points_{level}")
    rows = []
    for feature in payload["features"]:
        code = feature["properties"]["id"]
        if code[:2] in countries:
            lon, lat = feature["geometry"]["coordinates"]
            rows.append({"region": code, "name": feature["properties"].get("na", code), "country": code[:2], "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


def _parse_week(code):
    year, week = code.split("-W")
    return int(year), int(week)


def _deaths_frame(frame):
    if frame.empty:
        return pd.DataFrame(columns=["region", "iso_year", "iso_week", "deaths"])
    parsed = frame["time"].map(_parse_week)
    return pd.DataFrame(
        {
            "region": frame["geo"].values,
            "iso_year": [p[0] for p in parsed],
            "iso_week": [p[1] for p in parsed],
            "deaths": frame["value"].values,
        }
    )


def weekly_deaths(regions, children, since, cache, log):
    frames = []
    for region in regions:
        try:
            frame = _deaths_frame(eurostat("demo_r_mwk3_ts", {"sex": "T", "unit": "NR", "geo": region, "sinceTimePeriod": since}, cache))
        except SourceError:
            frame = pd.DataFrame()
        if len(frame) > 0:
            frames.append(frame)
            log.append({"region": region, "deaths_source": "NUTS 2 series"})
            continue
        parts = []
        for child in children.get(region, []):
            try:
                part = _deaths_frame(eurostat("demo_r_mwk3_ts", {"sex": "T", "unit": "NR", "geo": child, "sinceTimePeriod": since}, cache))
            except SourceError:
                part = pd.DataFrame()
            if len(part) > 0:
                parts.append(part)
        if parts and len(parts) == len(children.get(region, [])):
            merged = pd.concat(parts)
            counts = merged.groupby(["iso_year", "iso_week"])["region"].nunique()
            complete = counts[counts == len(parts)].index
            summed = merged.groupby(["iso_year", "iso_week"])["deaths"].sum().loc[complete].reset_index()
            summed["region"] = region
            frames.append(summed[["region", "iso_year", "iso_week", "deaths"]])
            log.append({"region": region, "deaths_source": f"sum of {len(parts)} NUTS 3 series"})
        else:
            log.append({"region": region, "deaths_source": "unavailable"})
    if not frames:
        raise SourceError("no weekly deaths could be retrieved")
    return pd.concat(frames, ignore_index=True)


def latest_value(code, filters, regions, column, cache, log):
    values = {}
    for region in regions:
        try:
            frame = eurostat(code, {**filters, "geo": region, "lastTimePeriod": 3}, cache)
        except SourceError:
            frame = pd.DataFrame()
        if len(frame) > 0:
            values[region] = float(frame.sort_values("time")["value"].iloc[-1])
    log.append({"covariate": column, "source": code, "regions_with_value": len(values), "regions_requested": len(regions)})
    return pd.Series(values, name=column, dtype=float)


def daily_tmax(points, start, end, cache):
    frames = []
    for row in points.itertuples():
        query = urllib.parse.urlencode(
            {
                "latitude": f"{row.lat:.4f}",
                "longitude": f"{row.lon:.4f}",
                "start_date": start,
                "end_date": end,
                "daily": "temperature_2m_max",
                "timezone": "GMT",
            }
        )
        payload = fetch_json(f"{OPEN_METEO}?{query}", cache, f"openmeteo_{row.region}_{start}_{end}", pause=0.5)
        daily = payload["daily"]
        frames.append(pd.DataFrame({"region": row.region, "date": pd.to_datetime(daily["time"]), "tmax": np.asarray(daily["temperature_2m_max"], dtype=float)}))
    return pd.concat(frames, ignore_index=True)


def load_real(config, cache):
    countries = list(config["countries"])
    log = []
    regions = nuts_points(2, countries, cache)
    exclude = set(config.get("exclude_regions", []))
    regions = regions[~regions["region"].isin(exclude)].reset_index(drop=True)
    level3 = nuts_points(3, countries, cache)
    children = {code: sorted(level3.loc[level3["region"].str[:4] == code, "region"]) for code in regions["region"]}
    deaths = weekly_deaths(list(regions["region"]), children, config["since"], cache, log)
    regions = regions[regions["region"].isin(deaths["region"].unique())].reset_index(drop=True)
    share65 = latest_value("demo_r_pjanind2", {"indic_de": "PC_Y65_MAX"}, list(regions["region"]), "share65", cache, log)
    gdp = latest_value("nama_10r_2gdp", {"unit": "PPS_EU27_2020_HAB"}, list(regions["region"]), "gdp_pps", cache, log)
    covariates = pd.DataFrame({"region": regions["region"]})
    covariates["share65"] = covariates["region"].map(share65)
    covariates["gdp_pps"] = covariates["region"].map(gdp)
    temps = daily_tmax(regions, config["temperature_start"], config["temperature_end"], cache)
    return {"regions": regions, "deaths": deaths, "temps": temps, "covariates": covariates, "log": log}
