import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_fixture_bundle.py"


def build(out: Path) -> None:
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)], check=True)


def test_build_is_byte_identical(tmp_path):
    build(tmp_path / "a")
    build(tmp_path / "b")
    for name in ("manifest.json", "graph.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_fixture_contents(tmp_path):
    build(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    graph = json.loads((tmp_path / "graph.json").read_text(encoding="utf-8"))
    assert manifest["synthetic_fixture"] is True
    assert manifest["display_warning"]
    assert manifest["model"]["revision"] is None
    assert len(graph["nodes"]) == 144
    assert graph["observations"] == []
    assert graph["edges"] == []
