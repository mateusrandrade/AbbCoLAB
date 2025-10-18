
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
from jiwer import wer, cer
from .config import EvalConfig
from .utils import discover_images, base_for_image, read_text_if_exists, append_csv, ensure_parent

PER_PAGE_FIELDS = ["doc_id","candidate_key","cer","wer"]
SUMMARY_FIELDS = ["engine","psm","count","cer_mean","wer_mean"]

def parse_key(key: str):
    if key.startswith("tess_psm"):
        return "tesseract", key.split("tess_psm")[-1]
    return key, ""

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

def eval_collection(cfg: EvalConfig) -> Dict[str, Any]:
    input_dir = Path(cfg.input_dir).resolve()
    files = discover_images(input_dir, cfg.glob)
    out_dir = Path(cfg.out_dir)
    ensure_parent(out_dir / "dummy")

    per_page = out_dir / "eval_by_page.csv"
    summary = out_dir / "eval_summary_by_engine_psm.csv"

    rows_page: List[Dict[str, Any]] = []
    agg: Dict[tuple, List[tuple]] = {}

    for img in files:
        base = base_for_image(img)
        curator = base.with_suffix(cfg.gold_suffix)
        curator_text = read_text_if_exists(curator)
        if not curator_text:
            continue
        cands = list_candidates_for_base(base)
        for key, path in cands.items():
            cand_text = read_text_if_exists(path) or ""
            _wer = float(wer(curator_text, cand_text)) if cand_text else 1.0
            _cer = float(cer(curator_text, cand_text)) if cand_text else 1.0
            rows_page.append({"doc_id": base.name, "candidate_key": key, "cer": _cer, "wer": _wer})
            eng, psm = parse_key(key)
            agg.setdefault((eng, psm), []).append((_cer, _wer))

    for r in rows_page:
        append_csv(per_page, PER_PAGE_FIELDS, r)

    for (eng, psm), vals in agg.items():
        cer_mean = sum(v[0] for v in vals) / len(vals)
        wer_mean = sum(v[1] for v in vals) / len(vals)
        append_csv(summary, SUMMARY_FIELDS, {
            "engine": eng, "psm": psm, "count": len(vals),
            "cer_mean": round(cer_mean,4), "wer_mean": round(wer_mean,4)
        })

    return {"pages_eval": len(rows_page), "groups": len(agg), "out_dir": str(out_dir)}
