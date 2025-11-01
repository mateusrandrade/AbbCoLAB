from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from daa_cli.config import ExportConfig
from daa_cli.export import export_dataset, fuse_candidates


def _prepare_sample_page(tmp_path: Path) -> Path:
    image = tmp_path / "page01.jpg"
    image.write_bytes(b"fake-image")

    curator = image.with_suffix(".curator.txt")
    curator.write_text("Texto correto", encoding="utf-8")

    image.with_suffix(".paddle.txt").write_text("Texto correto", encoding="utf-8")
    image.with_suffix(".tess.psm03.txt").write_text("Texte errado", encoding="utf-8")

    return image


def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_export_concat_mode_includes_all_candidates(tmp_path):
    _prepare_sample_page(tmp_path)

    cfg = ExportConfig(
        input_dir=str(tmp_path),
        glob="*.jpg",
        out=str(tmp_path / "dataset_concat.jsonl"),
        multi_hyp="concat",
    )

    result = export_dataset(cfg)

    dataset_rows = _read_jsonl(Path(result["out"]))
    manifest_rows = _read_jsonl(Path(result["manifest_jsonl"]))

    assert len(dataset_rows) == 1
    assert "<paddle>" in dataset_rows[0]["input_text"]
    assert "<tess psm=03>" in dataset_rows[0]["input_text"]
    assert dataset_rows[0]["meta"]["multi_hyp_mode"] == "concat"
    assert dataset_rows[0]["meta"]["selected_candidates"] == ["paddle", "tess_psm03"]

    assert manifest_rows[0]["multi_hyp_mode"] == "concat"
    assert manifest_rows[0]["selected_candidates"] == "paddle;tess_psm03"
    assert manifest_rows[0]["input_len"] == len(dataset_rows[0]["input_text"])


def test_export_best_mode_uses_lowest_cer_candidate(tmp_path):
    _prepare_sample_page(tmp_path)

    cfg = ExportConfig(
        input_dir=str(tmp_path),
        glob="*.jpg",
        out=str(tmp_path / "dataset_best.jsonl"),
        multi_hyp="best",
    )

    result = export_dataset(cfg)

    dataset_rows = _read_jsonl(Path(result["out"]))
    manifest_rows = _read_jsonl(Path(result["manifest_jsonl"]))

    assert len(dataset_rows) == 1
    assert dataset_rows[0]["input_text"] == "Texto correto"
    assert dataset_rows[0]["meta"]["selected_candidates"] == ["paddle"]
    assert dataset_rows[0]["meta"]["multi_hyp_mode"] == "best"

    assert manifest_rows[0]["multi_hyp_mode"] == "best"
    assert manifest_rows[0]["selected_candidates"] == "paddle"
    assert manifest_rows[0]["input_len"] == len(dataset_rows[0]["input_text"])


def test_export_fuse_mode_merges_candidates(tmp_path):
    _prepare_sample_page(tmp_path)

    cfg = ExportConfig(
        input_dir=str(tmp_path),
        glob="*.jpg",
        out=str(tmp_path / "dataset_fuse.jsonl"),
        multi_hyp="fuse",
    )

    result = export_dataset(cfg)

    dataset_rows = _read_jsonl(Path(result["out"]))
    manifest_rows = _read_jsonl(Path(result["manifest_jsonl"]))

    assert len(dataset_rows) == 1
    assert dataset_rows[0]["input_text"] == "Texto correto"
    assert dataset_rows[0]["meta"]["selected_candidates"] == ["paddle", "tess_psm03"]
    assert dataset_rows[0]["meta"]["multi_hyp_mode"] == "fuse"

    assert manifest_rows[0]["multi_hyp_mode"] == "fuse"
    assert manifest_rows[0]["selected_candidates"] == "paddle;tess_psm03"
    assert manifest_rows[0]["input_len"] == len(dataset_rows[0]["input_text"])


def test_fuse_candidates_prefers_anchor_on_ties():
    candidates = {
        "paddle": "lago",
        "tess_psm03": "lage",
    }

    fused = fuse_candidates(candidates, anchor_key="paddle")

    assert fused == "lago"


def test_fuse_candidates_handles_misaligned_lengths():
    candidates = {
        "paddle": "numero 123",
        "tess_psm03": "numero123",
        "easy": "nume ro 12 3",
    }

    fused = fuse_candidates(candidates, anchor_key="paddle")

    assert fused == "numero 123"


def test_fuse_candidates_single_candidate_fast_path():
    candidates = {
        "paddle": " texto\ncom  espaços  ",
    }

    fused = fuse_candidates(candidates)

    assert fused == "texto\ncom espaços"

    assert fused != "numero 123"
