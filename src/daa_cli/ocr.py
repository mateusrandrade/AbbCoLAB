
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from .config import OCRConfig
from .utils import discover_images, tesseract_version, sha256_of_file, append_csv, write_jsonl
from .backends import run_tesseract, run_easyocr, run_paddle

MANIFEST_FIELDS = [
    "timestamp","source_path","source_sha256",
    "engine","engine_version","device",
    "lang","oem","psm","format",
    "exit_code","duration_sec","stderr","out_path","notes"
]

def ocr_batch(cfg: OCRConfig) -> Dict[str, Any]:
    input_dir = Path(cfg.input_dir).resolve()
    files = discover_images(input_dir, cfg.glob)
    manifest_csv = input_dir / "manifests" / "ocr_manifest.csv"
    manifest_jsonl = input_dir / "manifests" / "ocr_manifest.jsonl"
    rows_jsonl = []
    stats = {"images": len(files), "rows": 0}

    tesseract_ver = None
    if "tesseract" in cfg.engines:
        tesseract_ver = tesseract_version()

    for img in files:
        sha = sha256_of_file(img)

        # Tesseract
        if "tesseract" in cfg.engines:
            rows = run_tesseract(img, cfg.lang, cfg.oem, cfg.psm, cfg.outputs, cfg.dry_run)
            for row in rows:
                append_csv(manifest_csv, MANIFEST_FIELDS, {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds")+"Z",
                    "source_path": str(img),
                    "source_sha256": sha,
                    "engine": "tesseract",
                    "engine_version": tesseract_ver,
                    "device": "cpu",
                    "lang": cfg.lang,
                    "oem": cfg.oem,
                    "psm": row.get("psm"),
                    "format": row.get("format"),
                    "exit_code": row.get("exit_code"),
                    "duration_sec": round(row.get("duration_sec", 0.0), 3),
                    "stderr": row.get("stderr",""),
                    "out_path": row.get("out_path"),
                    "notes": ""
                })
                rows_jsonl.append({
                    "engine": "tesseract",
                    "engine_version": tesseract_ver,
                    "device": "cpu",
                    "psm": row.get("psm"),
                    "format": row.get("format"),
                    "exit_code": row.get("exit_code"),
                    "duration_sec": row.get("duration_sec"),
                    "stderr": row.get("stderr"),
                    "out_path": row.get("out_path"),
                    "source_path": str(img),
                    "source_sha256": sha,
                })
                stats["rows"] += 1

        # PaddleOCR
        if "paddle" in cfg.engines:
            res = run_paddle(img, gpu=cfg.gpu)
            available = res.get("available", False)
            append_csv(manifest_csv, MANIFEST_FIELDS, {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds")+"Z",
                "source_path": str(img),
                "source_sha256": sha,
                "engine": "paddle",
                "engine_version": "",
                "device": "cuda" if cfg.gpu else "cpu",
                "lang": cfg.lang,
                "oem": "",
                "psm": "",
                "format": "txt",
                "exit_code": 0 if available else 1,
                "duration_sec": 0.0,
                "stderr": "" if available else res.get("error",""),
                "out_path": res.get("out_txt","") if available else "",
                "notes": "" if available else "backend ausente"
            })
            rows_jsonl.append({
                "engine": "paddle",
                "engine_version": "",
                "device": "cuda" if cfg.gpu else "cpu",
                "available": available,
                "out_txt": res.get("out_txt", ""),
                "out_json": res.get("out_json", ""),
                "error": res.get("error", "") if not available else "",
                "source_path": str(img),
                "source_sha256": sha
            })
            stats["rows"] += 1

        # EasyOCR
        if "easyocr" in cfg.engines:
            res = run_easyocr(img, langs=cfg.easyocr_langs, gpu=cfg.gpu)
            available = res.get("available", False)
            append_csv(manifest_csv, MANIFEST_FIELDS, {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds")+"Z",
                "source_path": str(img),
                "source_sha256": sha,
                "engine": "easyocr",
                "engine_version": "",
                "device": "cuda" if cfg.gpu else "cpu",
                "lang": ",".join(cfg.easyocr_langs),
                "oem": "",
                "psm": "",
                "format": "txt",
                "exit_code": 0 if available else 1,
                "duration_sec": 0.0,
                "stderr": "" if available else res.get("error",""),
                "out_path": res.get("out_txt","") if available else "",
                "notes": "" if available else "backend ausente"
            })
            rows_jsonl.append({
                "engine": "easyocr",
                "engine_version": "",
                "device": "cuda" if cfg.gpu else "cpu",
                "available": available,
                "out_txt": res.get("out_txt", ""),
                "out_json": res.get("out_json", ""),
                "error": res.get("error", "") if not available else "",
                "source_path": str(img),
                "source_sha256": sha
            })
            stats["rows"] += 1

    if rows_jsonl:
        write_jsonl(manifest_jsonl, rows_jsonl)

    return {"stats": stats, "manifest_csv": str(manifest_csv), "manifest_jsonl": str(manifest_jsonl)}
