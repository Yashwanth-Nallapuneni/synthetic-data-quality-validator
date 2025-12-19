import argparse
import json
import pandas as pd
import sys

from app.report import generate_trust_report


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Data Quality Validator"
    )

    parser.add_argument("--real", required=True, help="Path to real CSV")
    parser.add_argument("--synth", required=True, help="Path to synthetic CSV")
    parser.add_argument("--label", required=True, help="Label column name")
    parser.add_argument("--out", help="Output JSON file (optional)")

    args = parser.parse_args()

    # Load data
    real_df = pd.read_csv(args.real)
    synth_df = pd.read_csv(args.synth)

    if args.label not in real_df.columns:
        raise ValueError(f"Label column '{args.label}' not found in real data")

    if args.label not in synth_df.columns:
        raise ValueError(f"Label column '{args.label}' not found in synthetic data")

    # Run validator
    report = generate_trust_report(
        real_df,
        synth_df,
        label_column=args.label,
    )

    # Print warnings
    print("\n=== WARNINGS ===")
    if report["warnings"]:
        for w in report["warnings"]:
            print(f"- {w}")
    else:
        print("No major issues detected.")

    # Save report
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.out}")

    # Exit code (useful for CI)
    if report["warnings"]:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()