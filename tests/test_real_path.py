import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fmgcf_ewc import pipeline, sources
from fmgcf_ewc.synthetic import generate

ROOT = Path(__file__).resolve().parents[1]


def jsonstat(geo, times, values):
    present = [(t, v) for t, v in zip(times, values) if not np.isnan(v)]
    return {
        "id": ["geo", "time"],
        "size": [1, len(times)],
        "dimension": {
            "geo": {"category": {"index": {geo: 0}}},
            "time": {"category": {"index": {t: i for i, t in enumerate(times)}}},
        },
        "value": {str(times.index(t)): v for t, v in present},
    }


def fake_network(raw, missing_nuts2):
    regions = raw["regions"].copy()
    regions["code"] = [f"{c}{i + 1:02d}" for c, i in zip(regions["country"], regions.groupby("country").cumcount())]
    rename = dict(zip(regions["region"], regions["code"]))
    deaths = raw["deaths"].assign(region=raw["deaths"]["region"].map(rename))
    temps = raw["temps"].assign(region=raw["temps"]["region"].map(rename))
    covariates = raw["covariates"].assign(region=raw["covariates"]["region"].map(rename)).set_index("region")
    calls = []

    def respond(url, cache, key, pause=1.0):
        calls.append(url)
        if "nutspt_2" in url or "nutspt_3" in url:
            features = []
            for row in regions.itertuples():
                if "nutspt_2" in url:
                    features.append({"properties": {"id": row.code, "na": row.name}, "geometry": {"coordinates": [row.lon, row.lat]}})
                else:
                    for child in ("1", "2"):
                        features.append({"properties": {"id": row.code + child, "na": row.name}, "geometry": {"coordinates": [row.lon, row.lat]}})
            return {"features": features}
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        if "archive-api.open-meteo.com" in url:
            lat = float(query["latitude"][0])
            lon = float(query["longitude"][0])
            match = regions.loc[((regions["lat"] - lat) ** 2 + (regions["lon"] - lon) ** 2).idxmin(), "code"]
            block = temps[temps["region"] == match]
            return {"daily": {"time": block["date"].dt.strftime("%Y-%m-%d").tolist(), "temperature_2m_max": block["tmax"].round(2).tolist()}}
        dataset = parsed.path.rsplit("/", 1)[-1]
        geo = query["geo"][0]
        if dataset == "demo_r_mwk3_ts":
            parent = geo[:4]
            if geo == missing_nuts2:
                raise sources.SourceError("400")
            block = deaths[deaths["region"] == parent].sort_values(["iso_year", "iso_week"])
            times = [f"{y}-W{w:02d}" for y, w in zip(block["iso_year"], block["iso_week"])]
            values = block["deaths"].to_numpy(dtype=float)
            if len(geo) == 5:
                values = values / 2.0
            return jsonstat(geo, times, list(values))
        column = "share65" if dataset == "demo_r_pjanind2" else "gdp_pps"
        if geo not in covariates.index:
            raise sources.SourceError("400")
        return jsonstat(geo, ["2024"], [float(covariates.loc[geo, column])])

    return respond, calls


def test_real_data_path_runs_end_to_end_on_simulated_responses(monkeypatch, tmp_path):
    raw = generate(countries=("EL", "IT"), regions_per_country=7, seed=8)
    respond, calls = fake_network(raw, missing_nuts2="EL03")
    monkeypatch.setattr(sources, "fetch_json", respond)
    config = yaml.safe_load((ROOT / "configs" / "europe_south.yaml").read_text())
    config["data"]["countries"] = ["EL", "IT"]
    config["data"]["exclude_regions"] = ["IT07"]
    config["data"]["cache"] = str(tmp_path)
    config["model"]["rounds"] = 20
    config["alerts"]["bootstrap_samples"] = 30
    result = pipeline.run(config)
    log = {entry["region"]: entry["deaths_source"] for entry in result.source_log if "region" in entry}
    assert log["EL03"] == "sum of 2 NUTS 3 series"
    assert "IT07" not in set(result.vulnerability["region"])
    assert set(result.vulnerability["region"]) == {f"EL{i:02d}" for i in range(1, 8)} | {f"IT{i:02d}" for i in range(1, 7)}
    assert result.vulnerability["share65"].notna().all() and result.vulnerability["gdp_pps"].notna().all()
    assert sorted(result.vulnerability_info["covariates_used"]) == ["gdp_pps", "lat", "lon", "share65"]
    assert any("PPS_EU27_2020_HAB" in url for url in calls)
    assert any("PC_Y65_MAX" in url for url in calls)
    assert len(result.test_alerts) > 0
    assert set(result.models["model"]) == {"federated_fair", "federated_unconstrained", "federated_fair_dp", "centralised_fair"}
