from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(bold=True, color="FFFFFF")


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "n/a"
    return f"{value:.{digits}f}"


def _ci(row, metric):
    return f"{_fmt(row[metric])} (95% CI {_fmt(row[metric + '_ci_low'])} to {_fmt(row[metric + '_ci_high'])})"


def summary_rows(result):
    performance = result.performance.reset_index(drop=True)
    operational = result.models[result.models["operational"]].iloc[0]
    dp = result.privacy.get("federated_fair_dp", {})
    differences = result.differences
    info = result.vulnerability_info
    effects = ", ".join(f"{country} {value:+.4f}" for country, value in sorted(info.get("country_effects", {}).items()))
    rows = [
        ("Data source", result.config["data"]["source"]),
        ("Regions", int(result.vulnerability["region"].nunique())),
        ("Countries (federated clients)", ", ".join(sorted(result.vulnerability["country"].unique()))),
        ("Test years", ", ".join(str(y) for y in result.config["panel"]["test_years"])),
        ("Expected deaths", f"{result.config['panel'].get('baseline_method', 'trend')} over reference years, excluding the year evaluated"),
        ("Event definition", f"weekly deaths at least {result.config['panel']['event_threshold']:.0%} above expected, weeks {result.config['panel']['season_weeks'][0]}-{result.config['panel']['season_weeks'][1]}"),
        ("Operational model", operational["model"]),
        ("Test AUC (operational model)", _fmt(operational["test_auc"])),
        ("Vulnerability covariates", ", ".join(info["covariates_used"])),
        ("Country effect in vulnerability index", f"{info.get('country_effect', 'n/a')} (estimated effects: {effects})"),
        ("Fairness group", result.fairness_definition),
        ("Cost ratio (missed event / false alarm)", result.thresholds["cost_ratio"]),
        ("Warnings per strategy on validation", result.thresholds["validation_alerts_target"]),
        ("Selected vulnerability weighting (alpha)", f"{result.thresholds['alpha']:g}: {result.thresholds['reason']}"),
    ]
    for row in performance.itertuples(index=False):
        record = row._asdict()
        rows.append((f"Probability of detection, {record['strategy']}", _ci(record, "probability_of_detection")))
        rows.append((f"False alarm ratio, {record['strategy']}", _ci(record, "false_alarm_ratio")))
    for row in differences[differences["quantity"] == "probability of detection, high-vulnerability regions"].itertuples(index=False):
        rows.append((f"High-vulnerability detection gain, {row.comparison}", f"{_fmt(row.difference)} (95% CI {_fmt(row.ci_low)} to {_fmt(row.ci_high)})"))
    for row in differences[differences["quantity"] == "probability of detection, all regions"].itertuples(index=False):
        rows.append((f"Overall detection change, {row.comparison}", f"{_fmt(row.difference)} (95% CI {_fmt(row.ci_low)} to {_fmt(row.ci_high)})"))
    rows += [
        ("Differential privacy (DP variant)", f"epsilon {_fmt(dp.get('epsilon', float('nan')), 2)}, delta {dp.get('delta', 'n/a')}, noise multiplier {dp.get('noise_multiplier', 'n/a')}, clients {dp.get('clients', 'n/a')}"),
        ("Lead time", "one week ahead, using the week's observed temperature as a stand-in for a forecast"),
    ]
    return rows


def _format(sheet):
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    sheet.freeze_panes = "A2"
    for index, column in enumerate(sheet.columns, start=1):
        width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column)
        sheet.column_dimensions[get_column_letter(index)].width = min(max(12, width + 2), 80)


def write_excel(result, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    vulnerability = result.vulnerability[
        ["region", "name", "country", "share65", "gdp_pps", "heat_sensitivity", "country_effect", "sensitivity_within_country", "sensitivity_mean", "sensitivity_sd", "mvi", "mvi_tercile", "fair_group"]
    ].copy()
    sheets = {
        "Summary": pd.DataFrame(summary_rows(result), columns=["Item", "Value"]),
        "Alert performance": result.performance,
        "Vulnerability weighting": result.weighting,
        "Weighting vs uniform": result.differences,
        "By vulnerability tercile": result.by_tercile,
        "By fairness group": result.by_group,
        "Models": result.models,
        "Federated training": result.history,
        "Vulnerability index": vulnerability.sort_values("mvi", ascending=False),
        "Test alerts": result.test_alerts,
        "Regional graph": result.edges,
        "Data coverage": result.coverage,
        "Data sources": pd.DataFrame(result.source_log),
    }
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            _format(writer.sheets[name])
    return target


def markdown(result, config_path):
    lines = [f"## FMGCF-EWC early warning run: `{config_path}`", "", "| Item | Value |", "|---|---|"]
    for item, value in summary_rows(result):
        lines.append(f"| {item} | {value} |")
    lines += ["", "| Strategy | Alerts | Events | Hits | Misses | False alarms | POD | FAR | CSI |", "|---|---|---|---|---|---|---|---|---|"]
    for row in result.performance.itertuples():
        lines.append(
            f"| {row.strategy} | {row.alerts} | {row.events} | {row.hits} | {row.misses} | {row.false_alarms} | "
            f"{_fmt(row.probability_of_detection)} | {_fmt(row.false_alarm_ratio)} | {_fmt(row.critical_success_index)} |"
        )
    return "\n".join(lines) + "\n\n"
