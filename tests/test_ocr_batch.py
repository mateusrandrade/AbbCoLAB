from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from daa_cli.config import OCRConfig
from daa_cli import ocr as ocr_module


def _create_image(tmp_path: Path, name: str = "sample.jpg") -> Path:
    img_path = tmp_path / name
    img_path.write_bytes(b"fake-image-content")
    return img_path


def test_ocr_batch_uses_cached_tesseract_version(monkeypatch, tmp_path):
    _create_image(tmp_path)

    cfg = OCRConfig(
        input_dir=str(tmp_path),
        glob="*.jpg",
        engines=["tesseract"],
        outputs=["txt"],
        psm=[3],
        dry_run=True,
    )

    version_calls = []

    def fake_tesseract_version():
        version_calls.append("tesseract 5.0.0")
        return "tesseract 5.0.0"

    monkeypatch.setattr(ocr_module, "tesseract_version", fake_tesseract_version)

    fake_rows = [
        {
            "psm": 3,
            "format": "txt",
            "exit_code": 0,
            "duration_sec": 0.1,
            "stderr": "",
            "out_path": "out.txt",
        }
    ]

    monkeypatch.setattr(ocr_module, "run_tesseract", lambda *args, **kwargs: fake_rows)

    captured_manifest_rows = []

    def fake_append_csv(path, fields, row):
        captured_manifest_rows.append(row)

    monkeypatch.setattr(ocr_module, "append_csv", fake_append_csv)

    captured_json_rows = []

    def fake_write_jsonl(path, rows):
        captured_json_rows.extend(rows)

    monkeypatch.setattr(ocr_module, "write_jsonl", fake_write_jsonl)

    result = ocr_module.ocr_batch(cfg)

    assert len(version_calls) == 1, "tesseract_version should be called once per batch"
    assert captured_manifest_rows, "expected manifest rows to be recorded"
    assert all(
        row["engine_version"] == "tesseract 5.0.0" for row in captured_manifest_rows
    )
    assert result["stats"]["rows"] == len(captured_json_rows)

