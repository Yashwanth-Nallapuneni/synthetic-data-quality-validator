"""Command-line interface.

Exit codes:
  0  no warnings
  1  the report contains warnings
  2  usage or input error (bad arguments, unreadable file, missing label, column mismatch)
"""

import argparse
import json
import sys

import pandas as pd

from app.errors import InputError
from app.report import generate_trust_report

EXIT_CLEAN = 0
EXIT_WARNINGS = 1
EXIT_INPUT_ERROR = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synth-validate",
        description="Synthetic Data Quality Validator",
        epilog="Exit codes: 0 clean, 1 warnings found, 2 usage or input error.",
    )
    parser.add_argument("--real", required=True, help="Path to real CSV")
    parser.add_argument("--synth", required=True, help="Path to synthetic CSV")
    parser.add_argument("--label", required=True, help="Label column name")
    parser.add_argument("--out", help="Output JSON file (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)  # argparse exits with code 2 on usage errors

    try:
        real_df = pd.read_csv(args.real)
        synth_df = pd.read_csv(args.synth)
        report = generate_trust_report(real_df, synth_df, label_column=args.label, seed=args.seed)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        print(f"error: could not read input: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    print("\n=== WARNINGS ===")
    if report["warnings"]:
        for w in report["warnings"]:
            print(f"- {w}")
    else:
        print("No major issues detected.")

    if report["notices"]:
        print("\n=== NOTICES ===")
        for n in report["notices"]:
            print(f"- {n}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.out}")

    return EXIT_WARNINGS if report["warnings"] else EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
