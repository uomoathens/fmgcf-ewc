import json
from pathlib import Path

import yaml
from openpyxl import load_workbook

from fmgcf_ewc.cli import main

ROOT = Path(__file__).resolve().parents[1]

SHEETS = [
    "Summary",
    "Alert performance",
    "Vulnerability weighting",
    "Weighting vs uniform",
    "By vulnerability tercile",
    "By fairness group",
    "Models",
    "Federated training",
    "Vulnerability index",
    "Test alerts",
    "Regional graph",
    "Data coverage",
    "Data sources",
]


def small_config(tmp_path):
    config = yaml.safe_load((ROOT / "configs" / "synthetic.yaml").read_text())
    config["data"]["synthetic"]["regions_per_country"] = 6
    config["model"]["rounds"] = 30
    config["alerts"]["bootstrap_samples"] = 50
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_synthetic_run_produces_report_and_summary(tmp_path, capsys):
    config = small_config(tmp_path)
    summary = tmp_path / "summary.md"
    assert main(["run", "--config", str(config), "--output", str(tmp_path / "out"), "--summary", str(summary)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["regions"] == 24
    assert len(output["strategies"]) == 3
    assert output["selected_alpha"] in (0.0, 0.25, 0.5, 0.75, 1.0)
    workbook = load_workbook(tmp_path / "out" / "early_warning_report.xlsx")
    assert workbook.sheetnames == SHEETS
    assert workbook["Models"].max_row == 5
    text = summary.read_text(encoding="utf-8")
    assert "uniform threshold" in text
    assert "Selected vulnerability weighting" in text
    assert "Differential privacy" in text


def test_runs_are_reproducible(tmp_path, capsys):
    config = small_config(tmp_path)
    main(["run", "--config", str(config), "--output", str(tmp_path / "a")])
    first = json.loads(capsys.readouterr().out)
    main(["run", "--config", str(config), "--output", str(tmp_path / "b")])
    second = json.loads(capsys.readouterr().out)
    first.pop("report")
    second.pop("report")
    assert first == second
