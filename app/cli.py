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
from app.report import build_response, generate_trust_report
from app.thresholds import load_thresholds

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
    parser.add_argument(
        "--holdout",
        help="Optional CSV of real rows the generator never saw (privacy baseline). "
        "Without it the real data is split 50/50 for the baseline.",
    )
    parser.add_argument(
        "--thresholds",
        help="Optional JSON file overriding warning thresholds (see app/thresholds.py)",
    )
    parser.add_argument("--label", required=True, help="Label column name")
    parser.add_argument("--out", help="Output JSON file (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    return parser


def _print_summary(report: dict) -> None:
    """Print the headline number of each section (sections that were skipped are omitted)."""
    print("=== SUMMARY ===")
    dep, det, priv = report["dependence"], report["detectability"], report["privacy"]
    div, util = report["diversity"], report["utility"]
    if dep["status"] == "ok":
        s = dep["summary"]
        print(
            f"Dependence: mean abs difference {s['mean_abs_diff']:.3f}, "
            f"worst pair {s['worst_pair'][0]} / {s['worst_pair'][1]} ({s['max_abs_diff']:.3f})"
        )
    if det["status"] == "ok":
        print(f"Detectability (C2ST): AUC {det['auc_mean']:.3f} +/- {det['auc_std']:.3f}")
    if priv["status"] == "ok":
        print(
            f"Privacy ({priv['mode']}): {priv['share_closer_than_holdout_p5']:.1%} of synthetic "
            f"rows closer than holdout p5, exact-match rate {priv['exact_match_rate']:.1%}, "
            f"risk {priv['privacy_risk_level']}"
        )
    if div["coverage"] and div["coverage"]["value"] is not None:
        print(f"Coverage (k={div['coverage']['k']}): {div['coverage']['value']:.3f}")
    if util["status"] == "ok":
        for name, m in util["models"].items():
            if util["task"] == "regression":
                print(
                    f"Utility {name}: TRTR R2 {m['trtr']['r2']:.3f}, TSTR R2 {m['tstr']['r2']:.3f}"
                )
            else:
                print(
                    f"Utility {name}: TRTR accuracy {m['trtr']['accuracy']:.3f}, "
                    f"TSTR accuracy {m['tstr']['accuracy']:.3f}"
                )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)  # argparse exits with code 2 on usage errors

    try:
        real_df = pd.read_csv(args.real)
        synth_df = pd.read_csv(args.synth)
        holdout_df = pd.read_csv(args.holdout) if args.holdout else None
        report = generate_trust_report(
            real_df,
            synth_df,
            label_column=args.label,
            seed=args.seed,
            thresholds=load_thresholds(args.thresholds),
            holdout_df=holdout_df,
        )
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        print(f"error: could not read input: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    _print_summary(report)

    print("\n=== WARNINGS ===")
    if report["warnings"]:
        for w in report["warnings"]:
            print(f"- [{w['code']}] {w['message']}")
    else:
        print("No major issues detected.")

    if report["notices"]:
        print("\n=== NOTICES ===")
        for n in report["notices"]:
            print(f"- {n}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(build_response(report), f, indent=2)
        print(f"\nReport saved to: {args.out}")

    return EXIT_WARNINGS if report["warnings"] else EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
