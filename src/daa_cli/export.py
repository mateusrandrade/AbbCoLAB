
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
    "doc_id","source_image","num_candidates","has_curator","cer","wer","curator_len","input_len",
    "candidates_present","multi_hyp_mode","selected_candidates"
]

@dataclass
class Example:
    doc_id: str
    target_text: str
    candidates: Dict[str, str]
    tagged_candidates: Dict[str, str]
    meta: Dict[str, Any]

    def build_input(self, mode: str) -> Dict[str, Any]:
        mode_normalized = (mode or "").strip().lower()
        if mode_normalized == "concat":
            joined = "\n".join(
                self.tagged_candidates[key] for key in sorted(self.tagged_candidates.keys())
            ).strip()
            return {
                "input_text": joined,
                "selected_candidates": sorted(self.candidates.keys()),
            }

        if mode_normalized == "best":
            if not self.candidates:
                return {
                    "input_text": "",
                    "selected_candidates": [],
                }

            best_key: Optional[str] = None
            best_scores: Optional[tuple[float, float, int]] = None
            for key in sorted(self.candidates.keys()):
                text = self.candidates[key]
                cand_wer = float(wer(self.target_text, text)) if text else 1.0
                cand_cer = float(cer(self.target_text, text)) if text else 1.0
                score = (cand_cer, cand_wer, len(text or ""))
                if best_scores is None or score < best_scores:
                    best_scores = score
                    best_key = key

            assert best_key is not None  # for mypy
            return {
                "input_text": self.candidates[best_key],
                "selected_candidates": [best_key],
            }

        if mode_normalized == "fuse":
            raise ValueError("Modo multi_hyp='fuse' ainda não é suportado. Use concat ou best.")

        raise ValueError(
            "Modo multi_hyp='{mode}' inválido. Escolha entre concat ou best.".format(mode=mode)
        )

def make_example_for_image(img: Path, gold_suffix: str) -> Optional[Example]:
    base = base_for_image(img)
    curator = base.with_suffix(gold_suffix)
    curator_text = read_text_if_exists(curator)
    if curator_text is None:
        return None

    cands_paths = list_candidates_for_base(base)
    candidates: Dict[str, str] = {}
    tagged_candidates: Dict[str, str] = {}

    for key in sorted(cands_paths.keys()):
        txt = read_text_if_exists(cands_paths[key])
        if txt:
            candidates[key] = txt
            if key.startswith("tess_psm"):
                psm = key.split("tess_psm")[-1]
                tagged = f"<tess psm={psm}> {txt} </tess>"
            elif key == "paddle":
                tagged = f"<paddle> {txt} </paddle>"
            elif key == "easy":
                tagged = f"<easy> {txt} </easy>"
            else:
                tagged = txt

            tagged_candidates[key] = tagged

    meta = {
        "source_image": str(img),
        "candidates_keys": sorted(list(candidates.keys()))
    }
    return Example(
        doc_id=base.name,
        target_text=curator_text,
        candidates=candidates,
        tagged_candidates=tagged_candidates,
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

        try:
            input_info = ex.build_input(cfg.multi_hyp)
        except ValueError as exc:
            raise SystemExit(str(exc))

        input_text = input_info["input_text"]
        selected_candidates = input_info["selected_candidates"]

        _wer = float(wer(ex.target_text, input_text)) if input_text else 1.0
        _cer = float(cer(ex.target_text, input_text)) if input_text else 1.0

        meta = dict(ex.meta)
        meta["multi_hyp_mode"] = cfg.multi_hyp
        meta["selected_candidates"] = selected_candidates

        rows_export.append({
            "doc_id": ex.doc_id,
            "input_text": input_text,
            "target_text": ex.target_text,
            "candidates": ex.candidates,
            "meta": meta
        })

        rows_manifest.append({
            "doc_id": ex.doc_id,
            "source_image": ex.meta["source_image"],
            "num_candidates": len(ex.candidates),
            "has_curator": True,
            "cer": _cer,
            "wer": _wer,
            "curator_len": len(ex.target_text or ""),
            "input_len": len(input_text or ""),
            "candidates_present": ";".join(ex.meta["candidates_keys"]),
            "multi_hyp_mode": cfg.multi_hyp,
            "selected_candidates": ";".join(selected_candidates),
        })

    if cfg.fail_if_no_gold and found_curators == 0:
        raise SystemExit("Nenhum arquivo *.curator.txt encontrado na coleção. Export abortado.")

    write_jsonl(out_path, rows_export)
    ensure_parent(manifest_csv)
    for row in rows_manifest:
        append_csv(manifest_csv, EXPORT_FIELDS, row)
    write_jsonl(manifest_jsonl, rows_manifest)

    return {"items": len(rows_export), "out": str(out_path), "manifest_csv": str(manifest_csv), "manifest_jsonl": str(manifest_jsonl)}
