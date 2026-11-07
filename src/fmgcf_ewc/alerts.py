import numpy as np
import pandas as pd


def vulnerability_thresholds(mvi, cost_ratio):
    return 1.0 / (1.0 + cost_ratio * np.asarray(mvi, dtype=float))


def modulated_thresholds(mvi, cost_ratio, alpha, centre):
    weight = centre + float(alpha) * (np.asarray(mvi, dtype=float) - centre)
    return 1.0 / (1.0 + cost_ratio * weight)


def shift_for_volume(probabilities, thresholds, target_alerts):
    margins = np.asarray(probabilities) - np.asarray(thresholds)
    return matched_uniform_threshold(margins, target_alerts)


def apply(probabilities, thresholds, shift):
    return ((np.asarray(probabilities) - np.asarray(thresholds)) >= shift).astype(int)


def select_alpha(frame, alphas, cost_ratio, centre, target_alerts, tolerance):
    rows = []
    for alpha in alphas:
        thresholds = modulated_thresholds(frame["mvi"], cost_ratio, alpha, centre)
        shift = shift_for_volume(frame["probability"], thresholds, target_alerts)
        alerts = apply(frame["probability"], thresholds, shift)
        overall = scores(alerts, frame["event"])
        high = frame["mvi_tercile"] == "high"
        high_scores = scores(alerts[high.to_numpy()], frame.loc[high, "event"])
        rows.append(
            {
                "alpha": float(alpha),
                "shift": shift,
                "validation_alerts": overall["alerts"],
                "validation_pod_all": overall["probability_of_detection"],
                "validation_far_all": overall["false_alarm_ratio"],
                "validation_pod_high_vulnerability": high_scores["probability_of_detection"],
            }
        )
    table = pd.DataFrame(rows)
    reference = table.loc[table["alpha"] == 0.0, "validation_pod_all"].iloc[0]
    allowed = table[table["validation_pod_all"] >= reference - tolerance]
    best = allowed.sort_values(["validation_pod_high_vulnerability", "alpha"], ascending=[False, True]).iloc[0]
    table["selected"] = table["alpha"] == best["alpha"]
    table["passes_no_harm_check"] = table["validation_pod_all"] >= reference - tolerance
    if best["alpha"] == 0.0:
        reason = "no vulnerability weighting improved detection in high-vulnerability regions without lowering overall detection on validation; uniform threshold kept"
    else:
        reason = f"largest gain in high-vulnerability detection on validation among weightings that do not lower overall detection (tolerance {tolerance})"
    return float(best["alpha"]), float(best["shift"]), table, reason


def matched_uniform_threshold(probabilities, target_alerts):
    ordered = np.sort(np.asarray(probabilities))[::-1]
    if target_alerts <= 0:
        return float(ordered[0] + 1e-9) if len(ordered) else 1.0
    if target_alerts >= len(ordered):
        return float(ordered[-1])
    return float((ordered[target_alerts - 1] + ordered[target_alerts]) / 2.0)


def contingency(alerts, events):
    alerts = np.asarray(alerts).astype(bool)
    events = np.asarray(events).astype(bool)
    hits = int(np.sum(alerts & events))
    misses = int(np.sum(~alerts & events))
    false_alarms = int(np.sum(alerts & ~events))
    return hits, misses, false_alarms


def scores(alerts, events):
    hits, misses, false_alarms = contingency(alerts, events)
    pod = hits / (hits + misses) if hits + misses else np.nan
    far = false_alarms / (hits + false_alarms) if hits + false_alarms else np.nan
    csi = hits / (hits + misses + false_alarms) if hits + misses + false_alarms else np.nan
    return {
        "alerts": int(np.sum(alerts)),
        "events": int(np.sum(events)),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "probability_of_detection": pod,
        "false_alarm_ratio": far,
        "critical_success_index": csi,
        "miss_rate": 1 - pod if not np.isnan(pod) else np.nan,
    }


def bootstrap(frame, alert_column, statistic, samples=500, seed=0):
    rng = np.random.default_rng(seed)
    regions = frame["region"].unique()
    groups = {region: block for region, block in frame.groupby("region")}
    values = []
    for _ in range(int(samples)):
        drawn = rng.choice(regions, size=len(regions), replace=True)
        sample = pd.concat([groups[r] for r in drawn], ignore_index=True)
        values.append(statistic(sample, alert_column))
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))


def pod_of(sample, column):
    return scores(sample[column], sample["event"])["probability_of_detection"]


def far_of(sample, column):
    return scores(sample[column], sample["event"])["false_alarm_ratio"]


def csi_of(sample, column):
    return scores(sample[column], sample["event"])["critical_success_index"]


def pod_high_of(sample, column):
    block = sample[sample["mvi_tercile"] == "high"]
    return scores(block[column], block["event"])["probability_of_detection"]


def miss_gap_of(sample, column):
    a = sample[sample["fair_group"] == 1]
    b = sample[sample["fair_group"] == 0]
    return scores(a[column], a["event"])["miss_rate"] - scores(b[column], b["event"])["miss_rate"]


def difference_interval(frame, column_a, column_b, statistic, samples=500, seed=0):
    rng = np.random.default_rng(seed)
    regions = frame["region"].unique()
    groups = {region: block for region, block in frame.groupby("region")}
    values = []
    for _ in range(int(samples)):
        drawn = rng.choice(regions, size=len(regions), replace=True)
        sample = pd.concat([groups[r] for r in drawn], ignore_index=True)
        values.append(statistic(sample, column_a) - statistic(sample, column_b))
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.mean(values)), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
