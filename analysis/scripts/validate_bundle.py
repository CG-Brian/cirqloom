"""Validate a bundle directory. Exit code 1 on any problem (used by CI)."""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from cirqloom_analysis.export.bundle_writer import read_bundle
from cirqloom_analysis.schema.validation import BundleValidationError, validate_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    args = parser.parse_args()
    try:
        manifest, graph = read_bundle(args.bundle_dir)
        validate_bundle(manifest, graph)
    except (OSError, ValidationError, BundleValidationError) as exc:
        print(f"INVALID {args.bundle_dir}\n{exc}", file=sys.stderr)
        return 1
    print(f"OK {args.bundle_dir}: {len(graph.nodes)} nodes, {len(graph.observations)} observations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
