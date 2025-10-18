
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from jiwer import wer, cer
from .config import ExportConfig
from .utils import discover_images, base_for_image, read_text_if_exists, write_jsonl, append_csv, ensure_parent

def list_candidates_for_base(base: Path) -> Dict[str, Path]:
    out = {}
    for cand in base.parent.glob(base.name + ".tess.psm??.txt"):
        psm = cand.stem.split("psm")[-1]
        out[f"tess_psm{psm}"] = cand
    p = base.with_suffix(".paddle.txt")
    if p.exists(): out["paddle"] = p
    e = base.with_suffix(".easy.txt")
    if e.exists(): out["easy"] = e
    return out

EXPORT_FIELDS = [
    "doc_id","source_image","num_candidates","has_curator","cer","wer","curator_len","input_len","candidates_present"
]

@dataclass
class Example:
    doc_id: str
    input_text: str
    target_text: str
    candidates: Dict[str, str]
    meta: Dict[str, Any]

def make_example_for_image(img: Path, gold_suffix: str) -> Optional[Example]:
    base = base_for_image(img)
    curator = base.with_suffix(gold_suffix)
    curator_text = read_text_if_exists(curator)
    if curator_text is None:
        return None

    cands_paths = list_candidates_for_base(base)
    candidates: Dict[str, str] = {}
    parts: List[str] = []

    for key in sorted(cands_paths.keys()):
        txt = read_text_if_exists(cands_paths[key])
        if txt:
            candidates[key] = txt
            if key.startswith("tess_psm"):
                psm = key.split("tess_psm")[-1]
                parts.append(f"<tess psm={psm}> {txt} </tess>")
            elif key == "paddle":
                parts.append(f"<paddle> {txt} </paddle>")
            elif key == "easy":
                parts.append(f"<easy> {txt} </easy>")

    input_text = "\n".join(parts).strip()

    meta = {
        "source_image": str(img),
        "candidates_keys": sorted(list(candidates.keys()))
    }
    return Example(
        doc_id=base.name,
        input_text=input_text,
        target_text=curator_text,
        candidates=candidates,
        meta=meta
    )

def export_dataset(cfg: ExportConfig) -> Dict[str, Any]:
    input_dir = Path(cfg.input_dir).resolve()
    files = discover_images(input_dir, cfg.glob)
    out_path = Path(cfg.out)
    manifest_csv = (out_path.parent / "export_manifest.csv")
    manifest_jsonl = (out_path.parent / "export_manifest.jsonl")

    rows_export: List[Dict[str, Any]] = []
    rows_manifest: List[Dict[str, Any]] = []

    found_curators = 0
    for img in files:
        ex = make_example_for_image(img, cfg.gold_suffix)
        if ex is None:
            continue
        found_curators += 1

        _wer = float(wer(ex.target_text, ex.input_text)) if ex.input_text else 1.0
        _cer = float(cer(ex.target_text, ex.input_text)) if ex.input_text else 1.0

        rows_export.append({
            "doc_id": ex.doc_id,
            "input_text": ex.input_text,
            "target_text": ex.target_text,
            "candidates": ex.candidates,
            "meta": ex.meta
        })

        rows_manifest.append({
            "doc_id": ex.doc_id,
            "source_image": ex.meta["source_image"],
            "num_candidates": len(ex.candidates),
            "has_curator": True,
            "cer": _cer,
            "wer": _wer,
            "curator_len": len(ex.target_text or ""),
            "input_len": len(ex.input_text or ""),
            "candidates_present": ";".join(ex.meta["candidates_keys"])
        })

    if cfg.fail_if_no_gold and found_curators == 0:
        raise SystemExit("Nenhum arquivo *.curator.txt encontrado na coleção. Export abortado.")

    write_jsonl(out_path, rows_export)
    ensure_parent(manifest_csv)
    for row in rows_manifest:
        append_csv(manifest_csv, EXPORT_FIELDS, row)
    write_jsonl(manifest_jsonl, rows_manifest)

    return {"items": len(rows_export), "out": str(out_path), "manifest_csv": str(manifest_csv), "manifest_jsonl": str(manifest_jsonl)}
