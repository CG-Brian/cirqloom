"""Capture GPT-2 Small attention on repeated-token prompts and write a measured bundle.

Requires the real-model dependencies: pip install -e "analysis[dev,model]". Downloads the
pinned GPT-2 weights (~500 MB) into the Hugging Face cache on first run.

    python analysis/scripts/build_gpt2_bundle.py
"""

import argparse
import importlib.metadata
import platform
from pathlib import Path

from cirqloom_analysis.capture.controls import compare, controls_markdown
from cirqloom_analysis.capture.runner import run_experiment
from cirqloom_analysis.detectors.registry import default_registry
from cirqloom_analysis.export.bundle_writer import (
    read_bundle,
    validate_bundle_files,
    write_bundle,
)
from cirqloom_analysis.export.measured import build_measured_bundle
from cirqloom_analysis.models.gpt2 import GPT2Adapter
from cirqloom_analysis.prompts.repeated import CONTROL_SUITE, REPEATED_SUITE, build_suite
from cirqloom_analysis.schema.validation import validate_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "bundles" / "gpt2-small-repeated-v1"
BUNDLE_ID = "gpt2-small:repeated-tokens:v1"
GENERATOR = "analysis/scripts/build_gpt2_bundle.py"
NOTES = (
    "Attention-pattern detectors on repeated random tokens, with a paired non-repeated "
    "control. All detector results are 'experimental': heuristic attention scores, not "
    "causal evidence. No circuits are included."
)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lengths", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--seeds", type=int, default=8, help="prompts per experiment")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    adapter = GPT2Adapter(device=args.device, seed=0)
    info = adapter.info
    registry = default_registry()
    seeds = list(range(args.seeds))

    runs, comparisons = [], []
    for repeat_len in args.lengths:
        measured = run_experiment(adapter, build_suite(REPEATED_SUITE, repeat_len, seeds, info),
                                  registry)
        control = run_experiment(adapter, build_suite(CONTROL_SUITE, repeat_len, seeds, info),
                                 registry)
        runs += [measured, control]
        comparisons += compare(measured, control, info.n_heads)

    environment = {
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "transformer-lens": _version("transformer-lens"),
        "device": args.device,
        "dtype": "float32",
        "prepend_bos": "true",
        "bos_token": f"{info.bos_token_id} (<|endoftext|>, used as BOS by convention)",
        "precision": "scores rounded to 6 decimals; attention assets float16",
    }
    manifest, graph, assets = build_measured_bundle(
        runs, info, adapter.token_strings, bundle_id=BUNDLE_ID, generator=GENERATOR,
        environment=environment, notes=NOTES,
    )
    readme = render_readme(manifest, controls_markdown(comparisons), args)
    write_bundle(manifest, graph, args.out, assets, readme)

    written_manifest, written_graph = read_bundle(args.out)
    validate_bundle(written_manifest, written_graph)
    validate_bundle_files(args.out, written_manifest)
    print(controls_markdown(comparisons))
    print(f"wrote {len(graph.observations)} observations and {len(assets)} assets to {args.out}")
    return 0


def render_readme(manifest, controls: str, args) -> str:
    env = "\n".join(f"- {k}: {v}" for k, v in sorted(manifest.provenance.environment.items()))
    experiments = "\n".join(
        f"- `{e.id}` ({e.kind}, seq {e.sequence_length}, {len(e.seeds)} prompts): {e.description}"
        for e in manifest.experiments
    )
    return f"""# {manifest.bundle_id}

Measured GPT-2 Small attention bundle. Regenerate with:

    python analysis/scripts/build_gpt2_bundle.py --lengths {' '.join(map(str, args.lengths))} \\
        --seeds {args.seeds} --device {args.device}

## What this is

{manifest.provenance.notes}

- Model: `{manifest.model.hf_id}` at revision `{manifest.model.revision}`
  ({manifest.model.layers} layers × {manifest.model.heads_per_layer} heads).
- Detectors: previous-token, first-token (BOS) and induction attention masks
  (`analysis/cirqloom_analysis/detectors/`). Scores are mean masked attention per prompt,
  averaged over prompts (n = prompts, not cells). Status is `experimental` for every score.
- Attention assets: one prompt per measurement experiment, float16
  `[layer, head, q, k]`. Weights from the GPT-2 model card license apply to the model; this
  bundle contains derived attention values only, no weights.

## Experiments

{experiments}

## Input-side control (recorded, not judged)

Same heads on repeated input and on the paired non-repeated control. A content-dependent
head should drop on the control; a positional head should not. These numbers are reported
as measured; no pass/fail threshold is applied.

{controls}
## Environment

{env}
"""


if __name__ == "__main__":
    raise SystemExit(main())
