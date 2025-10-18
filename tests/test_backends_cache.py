from __future__ import annotations

from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from daa_cli import backends


def _create_image(tmp_path: Path, name: str = "sample.png") -> Path:
    img_path = tmp_path / name
    img_path.write_bytes(b"fake-image-content")
    return img_path


def test_run_easyocr_reuses_cached_reader(monkeypatch, tmp_path):
    image = _create_image(tmp_path)

    instantiation_args = []

    class DummyReader:
        def __init__(self, langs, gpu=False):
            instantiation_args.append((tuple(langs), bool(gpu)))

        def readtext(self, *args, **kwargs):
            return [([[0, 0], [1, 1], [1, 0], [0, 1]], "hello", 0.9)]

    class DummyEasyOCR:
        Reader = DummyReader

    def fake_import(module: str):
        if module == "easyocr":
            return DummyEasyOCR
        return None

    monkeypatch.setattr(backends, "_safe_import", fake_import)

    backends.clear_easyocr_cache()

    result_one = backends.run_easyocr(image, ["pt"], gpu=True)
    result_two = backends.run_easyocr(image, ["pt"], gpu=True)

    assert instantiation_args == [(("pt",), True)], "expected cached easyocr.Reader instance"
    assert result_one["available"] is True
    assert result_two["available"] is True

    backends.clear_easyocr_cache()
    backends.run_easyocr(image, ["pt"], gpu=True)

    assert len(instantiation_args) == 2, "cache clear should force new easyocr.Reader"

    backends.clear_easyocr_cache()


def test_run_paddle_reuses_cached_instance(monkeypatch, tmp_path):
    image = _create_image(tmp_path, "sample_paddle.png")

    instantiation_args = []

    class DummyPaddleOCR:
        def __init__(self, *, use_angle_cls, use_gpu, lang):
            instantiation_args.append((bool(use_gpu), lang))

        def ocr(self, *args, **kwargs):
            line = (
                [
                    [0, 0],
                    [1, 1],
                    [1, 0],
                    [0, 1],
                ],
                ("olá", 0.95),
            )
            return [[line]]

    class DummyPaddleModule:
        PaddleOCR = DummyPaddleOCR

    def fake_import(module: str):
        if module == "paddleocr":
            return DummyPaddleModule
        return None

    monkeypatch.setattr(backends, "_safe_import", fake_import)

    backends.clear_paddle_cache()

    result_one = backends.run_paddle(image, gpu=True)
    result_two = backends.run_paddle(image, gpu=True)

    assert instantiation_args == [(True, "pt")], "expected cached PaddleOCR instance"
    assert result_one["available"] is True
    assert result_two["available"] is True

    backends.clear_paddle_cache()
    backends.run_paddle(image, gpu=True)

    assert len(instantiation_args) == 2, "cache clear should force new PaddleOCR"

    backends.clear_paddle_cache()
