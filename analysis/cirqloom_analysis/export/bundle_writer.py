"""Deterministic bundle serialization. Validates before writing anything."""

import json
import shutil
from pathlib import Path, PurePosixPath

from pydantic import BaseModel

from cirqloom_analysis.schema.bundle import Graph, Manifest, TokensFile
from cirqloom_analysis.schema.validation import BundleValidationError, validate_bundle

MANIFEST_FILENAME = "manifest.json"
ASSETS_DIR = "assets"
README_FILENAME = "README.md"


def _dumps(model: BaseModel) -> str:
    """Serializes a Pydantic model into a consistently formatted JSON string."""

    payload = model.model_dump(mode="json")
    return json.dumps(payload,
                      indent=2,
                      sort_keys=True,
                      ensure_ascii=False,
                      allow_nan=False,) + "\n"


def _check_asset_path(relative: str) -> None:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.parts[:1] != (ASSETS_DIR,):
        raise ValueError(f"asset path must be inside {ASSETS_DIR}/, got {relative!r}")


def write_bundle(manifest: Manifest,
                 graph: Graph,
                 out_dir: Path,
                 assets: dict[str, bytes] | None = None,
                 readme: str | None = None) -> None:
    """Validates and writes the manifest, graph, asset files and optional README.

    The assets/ folder is generated content: it is replaced as a whole, so files from an
    earlier build with different experiments cannot linger.
    """

    validate_bundle(manifest, graph)
    assets = assets or {}
    for relative in assets:
        _check_asset_path(relative)
    referenced = {e.tokens_uri for e in manifest.experiments} | {
        e.attention.uri for e in manifest.experiments if e.attention is not None
    }
    missing = sorted(referenced - assets.keys())
    if missing:
        raise BundleValidationError([f"manifest references asset {m!r} that is not written"
                                     for m in missing])

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out_dir / ASSETS_DIR, ignore_errors=True)
    for relative, content in sorted(assets.items()):
        target = out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    if readme is not None:
        (out_dir / README_FILENAME).write_text(readme, encoding="utf-8")
    (out_dir / MANIFEST_FILENAME).write_text(_dumps(manifest),
                                             encoding="utf-8")
    (out_dir / manifest.graph_uri).write_text(_dumps(graph),
                                             encoding="utf-8")


def read_bundle(bundle_dir: Path) -> tuple[Manifest, Graph]:
    """Loads and parses the manifest and graph JSON files into Pydantic models."""

    manifest_text = (bundle_dir /
                     MANIFEST_FILENAME).read_text(encoding="utf-8")
    manifest = Manifest.model_validate_json(manifest_text)
    graph_text = (bundle_dir /
                     manifest.graph_uri).read_text(encoding="utf-8")
    return manifest, Graph.model_validate_json(graph_text)


def validate_bundle_files(bundle_dir: Path, manifest: Manifest) -> None:
    """Checks the asset files a manifest points to: present, parseable, the right size."""

    problems: list[str] = []
    for exp in manifest.experiments:
        tokens_path = bundle_dir / exp.tokens_uri
        if not tokens_path.is_file():
            problems.append(f"experiment {exp.id!r}: missing {exp.tokens_uri}")
        else:
            tokens = TokensFile.model_validate_json(tokens_path.read_text(encoding="utf-8"))
            if tokens.experiment_id != exp.id:
                problems.append(f"{exp.tokens_uri}: belongs to {tokens.experiment_id!r}")
            if [p.seed for p in tokens.prompts] != exp.seeds:
                problems.append(f"{exp.tokens_uri}: prompt seeds differ from the manifest")
            for prompt in tokens.prompts:
                if len(prompt.token_ids) != exp.sequence_length:
                    problems.append(f"{exp.tokens_uri}: {prompt.id} has the wrong length")
                if len(prompt.tokens) != len(prompt.token_ids):
                    problems.append(f"{exp.tokens_uri}: {prompt.id} tokens/ids length differ")
        if exp.attention is not None:
            path = bundle_dir / exp.attention.uri
            size = 2
            for dim in exp.attention.shape:
                size *= dim
            if not path.is_file():
                problems.append(f"experiment {exp.id!r}: missing {exp.attention.uri}")
            elif path.stat().st_size != size:
                problems.append(
                    f"{exp.attention.uri}: {path.stat().st_size} bytes, expected {size}"
                )
    if problems:
        raise BundleValidationError(problems)
