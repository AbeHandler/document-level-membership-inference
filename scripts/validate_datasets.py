#!/usr/bin/env python3
"""
Validate that required datasets exist in datasets.txt before running pipeline.

Usage:
    python scripts/validate_datasets.py --member blockeddocs --non-member project-gutenberg-extended
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate datasets exist in datasets.txt"
    )
    parser.add_argument(
        "--member",
        type=str,
        required=True,
        help="Member dataset name (e.g., 'blockeddocs')",
    )
    parser.add_argument(
        "--non-member",
        type=str,
        required=True,
        help="Non-member dataset name (e.g., 'project-gutenberg-extended')",
    )
    parser.add_argument(
        "--datasets-file",
        type=str,
        default="datasets.txt",
        help="Path to datasets.txt file",
    )
    return parser.parse_args()


def load_available_datasets(datasets_file):
    """Load available dataset names from datasets.txt."""
    datasets_path = Path(datasets_file)

    if not datasets_path.exists():
        raise FileNotFoundError(f"datasets.txt not found at {datasets_path}")

    available = set()
    with open(datasets_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Extract dataset name after the slash
                if "/" in line:
                    dataset_name = line.split("/")[-1]
                    available.add(dataset_name)

    return available


def main():
    args = parse_args()

    try:
        available = load_available_datasets(args.datasets_file)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Available datasets in {args.datasets_file}:")
    for dataset in sorted(available):
        print(f"  - {dataset}")
    print()

    # Validate member dataset
    if args.member not in available:
        print(f"ERROR: Member dataset '{args.member}' not found in {args.datasets_file}", file=sys.stderr)
        print(f"Available datasets: {', '.join(sorted(available))}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"✓ Member dataset '{args.member}' found")

    # Validate non-member dataset
    if args.non_member not in available:
        print(f"ERROR: Non-member dataset '{args.non_member}' not found in {args.datasets_file}", file=sys.stderr)
        print(f"Available datasets: {', '.join(sorted(available))}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"✓ Non-member dataset '{args.non_member}' found")

    print()
    print("All datasets validated successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
