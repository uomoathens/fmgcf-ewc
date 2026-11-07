import numpy as np
import pandas as pd
import pytest

from fmgcf_ewc import alerts as al
from fmgcf_ewc.federated import Client, TrainingSettings, auc, federated_train, loss_and_gradient, predict, zcdp_epsilon
from fmgcf_ewc.graph import knn_adjacency, propagate
from fmgcf_ewc.panel import build_panel, expected_deaths
from fmgcf_ewc.synthetic import generate
from fmgcf_ewc.vulnerability import fairness_group, heat_sensitivity, vulnerability_index


def test_expected_deaths_exclude_the_year_itself_and_need_enough_years():
    rows = []
    for year, value in ((2016, 100), (2017, 110), (2018, 120), (2019, 130)):
        rows.append({"region": "R1", "iso_year": year, "iso_week": 30, "deaths": float(value)})
    mean = expected_deaths(pd.DataFrame(rows), [2016, 2017, 2018, 2019], window=0, min_years=3, method="mean").set_index("iso_year")["expected"]
    assert mean[2019] == pytest.approx(110.0)
    assert mean[2016] == pytest.approx(120.0)
    trend = expected_deaths(pd.DataFrame(rows), [2016, 2017, 2018, 2019], window=0, min_years=3, method="trend").set_index("iso_year")["expected"]
    assert trend[2019] == pytest.approx(130.0)
    assert trend[2016] == pytest.approx(100.0)
    short = expected_deaths(pd.DataFrame(rows), [2016, 2017, 2018, 2019], window=0, min_years=4)
    assert short["expected"].isna().all()


def test_test_years_never_enter_the_baseline():
    raw = generate(regions_per_country=3, seed=4)
    config = {
        "reference_years": [2015, 2016, 2017, 2018, 2019, 2023, 2024, 2025],
        "train_years": [2015, 2016, 2017, 2018, 2019, 2023],
        "validation_years": [2024],
        "test_years": [2025],
        "event_threshold": 0.1,
        "season_weeks": [23, 38],
    }
    with_test = build_panel(raw, config)
    altered = {**raw, "deaths": raw["deaths"].assign(deaths=np.where(raw["deaths"]["iso_year"] == 2025, raw["deaths"]["deaths"] * 3, raw["deaths"]["deaths"]))}
    changed = build_panel(altered, config)
    earlier = with_test["iso_year"] < 2025
    assert np.allclose(with_test.loc[earlier, "expected"].to_numpy(), changed.loc[earlier, "expected"].to_numpy(), equal_nan=True)


def test_graph_is_symmetric_and_row_normalised():
    regions = pd.DataFrame({"region": ["A", "B", "C", "D"], "lat": [37.0, 37.1, 40.0, 40.1], "lon": [23.0, 23.1, 22.0, 22.1]})
    codes, operator, _ = knn_adjacency(regions, 1)
    assert np.allclose(operator.sum(axis=1), 1.0)
    assert (operator > 0).sum() == (operator.T > 0).sum()
    assert operator[0, 1] > 0 and operator[0, 2] == 0


def test_propagation_averages_over_self_and_neighbours():
    regions = pd.DataFrame({"region": ["A", "B"], "lat": [37.0, 37.1], "lon": [23.0, 23.1]})
    panel = pd.DataFrame({"region": ["A", "B"], "iso_year": [2025, 2025], "iso_week": [30, 30], "x": [2.0, 4.0]})
    out = propagate(panel, regions, ["x"], 1)
    assert out["x_neighbours"].tolist() == pytest.approx([3.0, 3.0])


def _sensitivity_rows(covariates, country_shift):
    rows = []
    rng = np.random.default_rng(0)
    for row in covariates.itertuples():
        slope = 0.01 + 0.003 * (row.share65 - 21.0) + country_shift.get(row.region[:2], 0.0)
        for _ in range(30):
            anomaly = rng.normal(0, 3)
            rows.append({"region": row.region, "temp_anomaly": anomaly, "excess": slope * anomaly + rng.normal(0, 0.005)})
    return heat_sensitivity(pd.DataFrame(rows))


def test_vulnerability_index_tracks_estimated_sensitivity():
    raw = generate(regions_per_country=10, seed=5)
    sensitivity = _sensitivity_rows(raw["covariates"], {})
    index, info = vulnerability_index(raw["regions"], raw["covariates"], sensitivity)
    assert index["mvi"].between(0, 1).all()
    assert np.corrcoef(index["mvi"], index["share65"])[0, 1] > 0.8
    assert set(index["mvi_tercile"]) == {"low", "medium", "high"}
    assert info["covariates_used"] == ["share65", "gdp_pps"]
    groups, definition = fairness_group(index, "low_income")
    assert groups.sum() == (index["gdp_pps"] < index["gdp_pps"].median()).sum()
    assert "GDP" in definition


def test_country_effect_is_separated_from_regional_vulnerability():
    raw = generate(regions_per_country=10, seed=5)
    sensitivity = _sensitivity_rows(raw["covariates"], {"AA": 0.05})
    removed, info = vulnerability_index(raw["regions"], raw["covariates"], sensitivity, settings={"country_effect": "remove"})
    kept, _ = vulnerability_index(raw["regions"], raw["covariates"], sensitivity, settings={"country_effect": "keep", "covariates": ["share65", "gdp_pps", "lat", "lon"]})
    assert info["country_effects"]["AA"] > 0.02
    share_removed = (removed.loc[removed["mvi_tercile"] == "high", "country"] == "AA").mean()
    share_kept = (kept.loc[kept["mvi_tercile"] == "high", "country"] == "AA").mean()
    assert share_removed < share_kept
    assert np.corrcoef(removed["mvi"], removed["share65"])[0, 1] > 0.7


def test_fairness_group_falls_back_when_income_is_missing():
    frame = pd.DataFrame({"country": ["A"] * 4, "gdp_pps": [np.nan] * 4, "share65": [18.0, 20.0, 22.0, 24.0], "mvi": [0.2, 0.4, 0.6, 0.8]})
    groups, definition = fairness_group(frame, "low_income")
    assert groups.tolist() == [0, 0, 1, 1]
    assert "65+" in definition


def test_missing_income_is_imputed_within_country_for_grouping():
    frame = pd.DataFrame({"country": ["A", "A", "B", "B", "B"], "gdp_pps": [10.0, 12.0, 30.0, 32.0, np.nan], "share65": [20.0] * 5, "mvi": [0.5] * 5})
    groups, definition = fairness_group(frame, "low_income")
    assert groups.tolist() == [1, 1, 0, 0, 0]
    assert "1 regions without GDP data" in definition


def test_miss_gap_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 3))
    y = (rng.random(200) < 0.4).astype(int)
    g = (rng.random(200) < 0.5).astype(int)
    settings = TrainingSettings(miss_weight=3.0, positive_weight=2.0, l2=0.01)
    theta = rng.normal(scale=0.3, size=4)
    _, gradient = loss_and_gradient(theta, x, y, g, settings)
    numeric = np.zeros(4)
    for i in range(4):
        step = np.zeros(4)
        step[i] = 1e-6
        numeric[i] = (loss_and_gradient(theta + step, x, y, g, settings)[0] - loss_and_gradient(theta - step, x, y, g, settings)[0]) / 2e-6
    assert np.allclose(gradient, numeric, atol=1e-5)


def test_zcdp_accounting():
    epsilon, rho = zcdp_epsilon(rounds=100, clients=4, noise_multiplier=2.0, delta=1e-5)
    assert rho == pytest.approx(100 / (2 * 4 * 4.0))
    assert epsilon == pytest.approx(rho + 2 * np.sqrt(rho * np.log(1e5)))
    assert zcdp_epsilon(10, 4, 0.0, 1e-5)[0] == float("inf")


def test_federated_training_learns_a_separable_signal():
    rng = np.random.default_rng(2)
    clients = []
    for name in ("AA", "BB", "CC"):
        x = rng.normal(size=(300, 2))
        y = (x[:, 0] + 0.3 * rng.normal(size=300) > 0.5).astype(int)
        clients.append(Client(name, x, y, (rng.random(300) < 0.5).astype(int)))
    result = federated_train(clients, 2, TrainingSettings(rounds=60, local_steps=5, learning_rate=0.5))
    x = np.vstack([c.features for c in clients])
    y = np.concatenate([c.labels for c in clients])
    assert auc(predict(result.parameters, x), y) > 0.9
    assert len(result.history) == 60


def test_vulnerability_thresholds_fall_as_vulnerability_rises():
    thresholds = al.vulnerability_thresholds([0.1, 0.5, 0.9], 4.0)
    assert thresholds[0] > thresholds[1] > thresholds[2]
    assert thresholds[1] == pytest.approx(1 / 3)


def test_matched_threshold_reproduces_the_alert_volume():
    probabilities = np.array([0.9, 0.8, 0.7, 0.4, 0.2])
    threshold = al.matched_uniform_threshold(probabilities, 3)
    assert int((probabilities >= threshold).sum()) == 3


def test_contingency_scores():
    result = al.scores([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert (result["hits"], result["misses"], result["false_alarms"]) == (2, 1, 1)
    assert result["probability_of_detection"] == pytest.approx(2 / 3)
    assert result["false_alarm_ratio"] == pytest.approx(1 / 3)
    assert result["critical_success_index"] == pytest.approx(0.5)


def test_region_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"region": np.repeat([f"R{i}" for i in range(20)], 10), "event": rng.integers(0, 2, 200), "alert": rng.integers(0, 2, 200)})
    low, high = al.bootstrap(frame, "alert", al.pod_of, samples=200)
    point = al.pod_of(frame, "alert")
    assert low <= point <= high


def test_weighting_selection_never_lowers_overall_detection():
    rng = np.random.default_rng(4)
    n = 600
    mvi = rng.uniform(0, 1, n)
    events = (rng.random(n) < np.where(mvi < 0.3, 0.5, 0.1)).astype(int)
    probability = np.clip(0.2 + 0.4 * events + rng.normal(0, 0.15, n), 0, 1)
    tercile = pd.qcut(pd.Series(mvi).rank(method="first"), 3, labels=["low", "medium", "high"]).astype(str)
    frame = pd.DataFrame({"mvi": mvi, "event": events, "probability": probability, "mvi_tercile": tercile})
    alpha, shift, table, reason = al.select_alpha(frame, [0.0, 0.5, 1.0], 4.0, float(mvi.mean()), 120, 0.0)
    reference = table.loc[table["alpha"] == 0.0, "validation_pod_all"].iloc[0]
    chosen = table.loc[table["alpha"] == alpha, "validation_pod_all"].iloc[0]
    assert chosen >= reference
    assert (table["validation_alerts"] == 120).all()
    assert table["selected"].sum() == 1


def test_weighting_selection_falls_back_to_uniform_when_weighting_hurts():
    rng = np.random.default_rng(5)
    n = 600
    mvi = rng.uniform(0, 1, n)
    events = (mvi < 0.2).astype(int)
    probability = np.clip(0.3 + 0.3 * events + rng.normal(0, 0.05, n), 0, 1)
    tercile = pd.qcut(pd.Series(mvi).rank(method="first"), 3, labels=["low", "medium", "high"]).astype(str)
    frame = pd.DataFrame({"mvi": mvi, "event": events, "probability": probability, "mvi_tercile": tercile})
    alpha, _, _, reason = al.select_alpha(frame, [0.0, 0.5, 1.0], 4.0, float(mvi.mean()), int(events.sum()), 0.0)
    assert alpha == 0.0
    assert "uniform threshold kept" in reason


def test_modulated_thresholds_reduce_to_a_single_threshold_without_weighting():
    thresholds = al.modulated_thresholds([0.1, 0.5, 0.9], 4.0, 0.0, 0.5)
    assert np.allclose(thresholds, 1 / 3)
    weighted = al.modulated_thresholds([0.1, 0.5, 0.9], 4.0, 1.0, 0.5)
    assert weighted[0] > weighted[1] > weighted[2]
