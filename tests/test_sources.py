import json

import numpy as np
import pandas as pd
import pytest

from fmgcf_ewc import sources
from fmgcf_ewc.jsonstat import to_frame


def payload(values):
    return {
        "id": ["sex", "geo", "time"],
        "size": [1, 2, 3],
        "dimension": {
            "sex": {"category": {"index": {"T": 0}}},
            "geo": {"category": {"index": {"EL30": 0, "EL43": 1}}},
            "time": {"category": {"index": {"2025-W27": 0, "2025-W28": 1, "2025-W29": 2}}},
        },
        "value": values,
    }


def test_jsonstat_sparse_dictionary_is_unflattened():
    frame = to_frame(payload({"0": 10, "4": 25, "5": 30}))
    assert frame.sort_values(["geo", "time"])[["geo", "time", "value"]].values.tolist() == [
        ["EL30", "2025-W27", 10.0],
        ["EL43", "2025-W28", 25.0],
        ["EL43", "2025-W29", 30.0],
    ]


def test_jsonstat_dense_list_skips_missing_cells():
    frame = to_frame(payload([1, None, 3, 4, 5, None]))
    assert len(frame) == 4
    assert frame.loc[frame["time"] == "2025-W29", "value"].tolist() == [3.0]


def test_cached_responses_are_reused(tmp_path, monkeypatch):
    calls = []

    class Response:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake(request, timeout):
        calls.append(request.full_url)
        return Response(json.dumps({"ok": True}).encode("utf-8"))

    monkeypatch.setattr(sources.urllib.request, "urlopen", fake)
    monkeypatch.setattr(sources.time, "sleep", lambda seconds: None)
    first = sources.fetch_json("https://example.org/a", tmp_path, "key")
    second = sources.fetch_json("https://example.org/a", tmp_path, "key")
    assert first == second == {"ok": True}
    assert len(calls) == 1


def test_missing_nuts2_series_is_rebuilt_from_complete_nuts3_children(tmp_path, monkeypatch):
    def fake(code, filters, cache):
        geo = filters["geo"]
        if geo == "EL30":
            return pd.DataFrame(columns=["geo", "time", "value"])
        weeks = ["2025-W27", "2025-W28"] if geo == "EL301" else ["2025-W27"]
        return pd.DataFrame({"geo": geo, "time": weeks, "value": [10.0] * len(weeks)})

    monkeypatch.setattr(sources, "eurostat", fake)
    log = []
    deaths = sources.weekly_deaths(["EL30"], {"EL30": ["EL301", "EL302"]}, "2025-W01", tmp_path, log)
    assert deaths[["region", "iso_year", "iso_week", "deaths"]].values.tolist() == [["EL30", 2025, 27, 20.0]]
    assert log[0]["deaths_source"] == "sum of 2 NUTS 3 series"


def test_daily_temperatures_are_parsed(tmp_path, monkeypatch):
    def fake(url, cache, key, pause=1.0):
        assert "temperature_2m_max" in url
        return {"daily": {"time": ["2025-07-01", "2025-07-02"], "temperature_2m_max": [36.1, None]}}

    monkeypatch.setattr(sources, "fetch_json", fake)
    points = pd.DataFrame({"region": ["EL30"], "lat": [38.0], "lon": [23.7]})
    frame = sources.daily_tmax(points, "2025-07-01", "2025-07-02", tmp_path)
    assert frame["tmax"].iloc[0] == pytest.approx(36.1)
    assert np.isnan(frame["tmax"].iloc[1])


def test_nuts_version_with_complete_series_is_selected(tmp_path, monkeypatch):
    codes = {2021: ["PT11", "PT16", "PT17"], 2024: ["PT11", "PT19", "PT1A", "PT1D"]}

    def points(level, countries, cache, version=2021):
        rows = [{"region": code if level == 2 else code + "1", "name": code, "country": "PT", "lat": 39.0, "lon": -8.0} for code in codes[version]]
        return pd.DataFrame(rows)

    def deaths(regions, children, since, cache, log):
        rows = []
        for region in regions:
            years = [2023, 2024, 2025] if region in ("PT11", "PT19", "PT1A", "PT1D") else [2023, 2024]
            for year in years:
                rows.append({"region": region, "iso_year": year, "iso_week": 30, "deaths": 100.0})
            log.append({"region": region, "deaths_source": "NUTS 2 series"})
        return pd.DataFrame(rows)

    monkeypatch.setattr(sources, "nuts_points", points)
    monkeypatch.setattr(sources, "weekly_deaths", deaths)
    log = []
    regions, frame = sources._country_series("PT", [2021, 2024], set(), "2023-W01", [2023, 2024, 2025], tmp_path, log)
    assert sorted(regions["region"]) == ["PT11", "PT19", "PT1A", "PT1D"]
    assert set(regions["nuts_version"]) == {2024}
    assert {"country": "PT", "nuts_version_selected": 2024} in log
    assert {"country": "PT", "nuts_version": 2021, "regions": 3, "regions_complete": 1} in log


def test_unchanged_nuts_codes_are_fetched_once(tmp_path, monkeypatch):
    calls = []

    def points(level, countries, cache, version=2021):
        return pd.DataFrame([{"region": "EL30" if level == 2 else "EL301", "name": "Attiki", "country": "EL", "lat": 38.0, "lon": 23.7}])

    def deaths(regions, children, since, cache, log):
        calls.append(tuple(regions))
        return pd.DataFrame([{"region": "EL30", "iso_year": 2025, "iso_week": 30, "deaths": 100.0}])

    monkeypatch.setattr(sources, "nuts_points", points)
    monkeypatch.setattr(sources, "weekly_deaths", deaths)
    regions, _ = sources._country_series("EL", [2021, 2024], set(), "2025-W01", [2025], tmp_path, [])
    assert calls == [("EL30",)]
    assert set(regions["nuts_version"]) == {2021}
