import argparse
import json
import sys
from pathlib import Path

import yaml

from .pipeline import run
from .report import markdown, write_excel


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fmgcf-ewc")
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--config", default="configs/synthetic.yaml")
    runner.add_argument("--output", default="outputs")
    runner.add_argument("--summary", default=None)
    arguments = parser.parse_args(argv)
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    result = run(config)
    output = Path(arguments.output)
    report = write_excel(result, output / "early_warning_report.xlsx")
    performance = result.performance.set_index("strategy")
    print(
        json.dumps(
            {
                "report": str(report),
                "regions": int(result.vulnerability["region"].nunique()),
                "test_events": int(performance.iloc[0]["events"]),
                "pod_uniform": performance.iloc[0]["probability_of_detection"],
                "pod_vulnerability_aware": performance.iloc[1]["probability_of_detection"],
                "far_uniform": performance.iloc[0]["false_alarm_ratio"],
                "far_vulnerability_aware": performance.iloc[1]["false_alarm_ratio"],
            },
            indent=2,
            default=float,
        )
    )
    if arguments.summary:
        with Path(arguments.summary).open("a", encoding="utf-8") as handle:
            handle.write(markdown(result, arguments.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
