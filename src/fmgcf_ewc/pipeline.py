from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import alerts as al
from .federated import Client, TrainingSettings, auc, centralised_train, federated_train, platt, predict
from .graph import edge_list, propagate
from .panel import build_panel, coverage, split
from .sources import load_real
from .synthetic import generate
from .vulnerability import fairness_group, heat_sensitivity, vulnerability_index

FEATURES = [
    "temp_anomaly",
    "tmax_peak",
    "temp_anomaly_previous",
    "excess_lagged",
    "temp_anomaly_neighbours",
    "excess_lagged_neighbours",
    "mvi",
]


@dataclass
class RunResult:
    config: dict
    coverage: pd.DataFrame
    vulnerability: pd.DataFrame
    vulnerability_info: dict
    fairness_definition: str
    models: pd.DataFrame
    history: pd.DataFrame
    performance: pd.DataFrame
    by_tercile: pd.DataFrame
    by_group: pd.DataFrame
    differences: pd.DataFrame
    test_alerts: pd.DataFrame
    edges: pd.DataFrame
    source_log: list
    thresholds: dict
    privacy: dict = field(default_factory=dict)


def load(config):
    data = config["data"]
    if data["source"] == "synthetic":
        return generate(**data.get("synthetic", {}))
    return load_real(data, data.get("cache", ".cache/ewc"))


def _standardise(parts, columns):
    train = parts["train"][columns]
    mean = train.mean()
    std = train.std().replace(0, 1.0)
    return {name: ((frame[columns] - mean) / std).fillna(0.0).to_numpy() for name, frame in parts.items()}


def _settings(model_cfg, positive_weight, **overrides):
    values = {
        "rounds": model_cfg["rounds"],
        "local_steps": model_cfg["local_steps"],
        "learning_rate": model_cfg["learning_rate"],
        "l2": model_cfg["l2"],
        "miss_weight": model_cfg["miss_weight"],
        "positive_weight": positive_weight,
        "clip": model_cfg.get("clip", 1.0),
        "noise_multiplier": 0.0,
        "delta": model_cfg.get("delta", 1e-5),
        "seed": model_cfg.get("seed", 0),
    }
    values.update(overrides)
    return TrainingSettings(**values)


def run(config):
    raw = load(config)
    panel = build_panel(raw, config["panel"])
    panel = propagate(panel, raw["regions"], ["temp_anomaly", "excess_lagged"], config["graph"]["neighbours"], config["graph"].get("hops", 1))
    parts = split(panel, config["panel"])
    sensitivity = heat_sensitivity(parts["train"])
    vulnerability, vulnerability_info = vulnerability_index(raw["regions"], raw["covariates"], sensitivity, config["model"].get("seed", 0))
    groups, definition = fairness_group(vulnerability, config["alerts"].get("fairness_group", "low_income"))
    vulnerability["fair_group"] = groups.values
    lookup = vulnerability.set_index("region")
    for name, frame in parts.items():
        for column in ("mvi", "mvi_tercile", "fair_group", "sensitivity_sd"):
            frame[column] = frame["region"].map(lookup[column])
        parts[name] = frame.dropna(subset=["mvi"]).reset_index(drop=True)
    matrices = _standardise(parts, FEATURES)
    train = parts["train"]
    labels = {name: frame["event"].to_numpy() for name, frame in parts.items()}
    group = {name: frame["fair_group"].to_numpy().astype(int) for name, frame in parts.items()}
    positives = max(1, int(labels["train"].sum()))
    positive_weight = (len(labels["train"]) - positives) / positives
    model_cfg = config["model"]
    clients = []
    for country, index in train.groupby("country").groups.items():
        rows = np.asarray(list(index))
        clients.append(Client(country, matrices["train"][rows], labels["train"][rows], group["train"][rows]))
    n_features = len(FEATURES)

    def monitor(parameters):
        return {"validation_auc": auc(predict(parameters, matrices["validation"]), labels["validation"])}

    variants = {
        "federated_fair": _settings(model_cfg, positive_weight),
        "federated_unconstrained": _settings(model_cfg, positive_weight, miss_weight=0.0),
        "federated_fair_dp": _settings(model_cfg, positive_weight, noise_multiplier=model_cfg.get("noise_multiplier", 1.0)),
    }
    trained = {}
    history = None
    privacy = {}
    for name, settings in variants.items():
        result = federated_train(clients, n_features, settings, monitor if name == model_cfg["operational"] else None)
        trained[name] = result.parameters
        privacy[name] = {"epsilon": result.epsilon, "rho": result.rho, "delta": settings.delta, "noise_multiplier": settings.noise_multiplier, "clip": settings.clip, "clients": len(clients), "rounds": settings.rounds}
        if name == model_cfg["operational"]:
            history = pd.DataFrame(result.history)
    trained["centralised_fair"] = centralised_train(matrices["train"], labels["train"], group["train"], n_features, variants["federated_fair"])
    privacy["centralised_fair"] = {"epsilon": float("nan"), "rho": float("nan"), "delta": float("nan"), "noise_multiplier": 0.0, "clip": float("nan"), "clients": 1, "rounds": variants["federated_fair"].rounds}
    model_rows = []
    for name, parameters in trained.items():
        row = {"model": name, "operational": name == model_cfg["operational"]}
        for split_name in ("train", "validation", "test"):
            row[f"{split_name}_auc"] = auc(predict(parameters, matrices[split_name]), labels[split_name])
        row.update({f"privacy_{k}": v for k, v in privacy[name].items()})
        model_rows.append(row)
    models = pd.DataFrame(model_rows)
    operational = trained[model_cfg["operational"]]
    calibrate = platt(predict(operational, matrices["validation"]), labels["validation"])
    probability = {name: calibrate(predict(operational, matrices[name])) for name in ("validation", "test")}
    cost_ratio = float(config["alerts"]["cost_ratio"])
    alerts_cfg = config["alerts"]
    for name in ("validation", "test"):
        frame = parts[name]
        frame["probability"] = probability[name]
        frame["threshold_vulnerability"] = al.vulnerability_thresholds(frame["mvi"], cost_ratio)
        frame["alert_vulnerability"] = (frame["probability"] >= frame["threshold_vulnerability"]).astype(int)
    target = int(parts["validation"]["alert_vulnerability"].sum())
    uniform = al.matched_uniform_threshold(parts["validation"]["probability"], target)
    for name in ("validation", "test"):
        parts[name]["alert_uniform"] = (parts[name]["probability"] >= uniform).astype(int)
    test = parts["test"]
    samples = alerts_cfg.get("bootstrap_samples", 500)
    rows = []
    for strategy, column in (("uniform threshold (matched alert volume)", "alert_uniform"), ("vulnerability-aware threshold", "alert_vulnerability")):
        entry = {"strategy": strategy, **al.scores(test[column], test["event"])}
        for label, statistic in (("probability_of_detection", al.pod_of), ("false_alarm_ratio", al.far_of), ("critical_success_index", al.csi_of)):
            low, high = al.bootstrap(test, column, statistic, samples, seed=1)
            entry[f"{label}_ci_low"] = low
            entry[f"{label}_ci_high"] = high
        rows.append(entry)
    performance = pd.DataFrame(rows)
    tercile_rows = []
    group_rows = []
    for strategy, column in (("uniform", "alert_uniform"), ("vulnerability-aware", "alert_vulnerability")):
        for tercile in ("high", "medium", "low"):
            block = test[test["mvi_tercile"] == tercile]
            tercile_rows.append({"strategy": strategy, "vulnerability_tercile": tercile, "regions": int(block["region"].nunique()), **al.scores(block[column], block["event"])})
        for value, label in ((1, "fairness group"), (0, "other regions")):
            block = test[test["fair_group"] == value]
            group_rows.append({"strategy": strategy, "group": label, "regions": int(block["region"].nunique()), **al.scores(block[column], block["event"])})
    differences = []
    for label, statistic in (
        ("probability of detection, high-vulnerability regions", al.pod_high_of),
        ("probability of detection, all regions", al.pod_of),
        ("false alarm ratio, all regions", al.far_of),
        ("miss-rate gap (fairness group minus others)", al.miss_gap_of),
    ):
        mean, low, high = al.difference_interval(test, "alert_vulnerability", "alert_uniform", statistic, samples, seed=2)
        differences.append({"quantity": label, "vulnerability_aware_minus_uniform": mean, "ci_low": low, "ci_high": high})
    keep = ["region", "country", "iso_year", "iso_week", "deaths", "expected", "excess", "event", "temp_anomaly", "mvi", "mvi_tercile", "fair_group", "probability", "threshold_vulnerability", "alert_vulnerability", "alert_uniform"]
    names = raw["regions"].set_index("region")["name"]
    test_alerts = test[keep].copy()
    test_alerts.insert(1, "name", test_alerts["region"].map(names))
    return RunResult(
        config=config,
        coverage=coverage(panel, parts),
        vulnerability=vulnerability,
        vulnerability_info=vulnerability_info,
        fairness_definition=definition,
        models=models,
        history=history,
        performance=performance,
        by_tercile=pd.DataFrame(tercile_rows),
        by_group=pd.DataFrame(group_rows),
        differences=pd.DataFrame(differences),
        test_alerts=test_alerts,
        edges=edge_list(raw["regions"], config["graph"]["neighbours"]),
        source_log=raw["log"],
        thresholds={"uniform": uniform, "cost_ratio": cost_ratio, "validation_alerts_target": target},
        privacy=privacy,
    )
