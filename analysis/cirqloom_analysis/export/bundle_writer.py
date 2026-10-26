"""Deterministic bundle serialization. Validates before writing anything."""

import json
from pathlib import Path

from pydantic import BaseModel

from cirqloom_analysis.schema.bundle import Graph, Manifest
from cirqloom_analysis.schema.validation import validate_bundle

MANIFEST_FILENAME = "manifest.json"


def _dumps(model: BaseModel) -> str:
    """Serializes a Pydantic model into a consistently formatted JSON string."""

    payload = model.model_dump(mode="json")
    return json.dumps(payload,
                      indent=2,
                      sort_keys=True,
                      ensure_ascii=False,
                      allow_nan=False,) + "\n"


def write_bundle(manifest: Manifest,
                 graph: Graph,
                 out_dir: Path) -> None:
    """Validates and writes the manifest and graph as JSON files."""

    validate_bundle(manifest, graph)
    out_dir.mkdir(parents=True, exist_ok=True)
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
