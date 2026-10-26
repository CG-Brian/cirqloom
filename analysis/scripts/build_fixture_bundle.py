"""Write the synthetic demo fixture to data/fixtures/gpt2-small-demo/."""

import argparse
from pathlib import Path

from cirqloom_analysis.export.bundle_writer import write_bundle
from cirqloom_analysis.export.fixture import build_fixture

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "fixtures" / "gpt2-small-demo"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest, graph = build_fixture()
    write_bundle(manifest, graph, args.out)
    print(f"wrote fixture bundle to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
