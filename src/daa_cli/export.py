
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import difflib
import re
import unicodedata
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

    def _select_best_candidate_key(self) -> Optional[str]:
        if not self.candidates:
            return None

        best_key: Optional[str] = None
        best_scores: Optional[Tuple[float, float, int]] = None
        for key in sorted(self.candidates.keys()):
            text = self.candidates[key]
            cand_wer = float(wer(self.target_text, text)) if text else 1.0
            cand_cer = float(cer(self.target_text, text)) if text else 1.0
            score = (cand_cer, cand_wer, len(text or ""))
            if best_scores is None or score < best_scores:
                best_scores = score
                best_key = key

        return best_key

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
            best_key = self._select_best_candidate_key()
            if best_key is None:
                return {
                    "input_text": "",
                    "selected_candidates": [],
                }

            return {
                "input_text": self.candidates[best_key],
                "selected_candidates": [best_key],
            }

        if mode_normalized == "fuse":
            if not self.candidates:
                return {
                    "input_text": "",
                    "selected_candidates": [],
                }

            anchor = self._select_best_candidate_key()
            fused = fuse_candidates(self.candidates, anchor_key=anchor)
            return {
                "input_text": fused,
                "selected_candidates": sorted(self.candidates.keys()),
            }

        raise ValueError(
            "Modo multi_hyp='{mode}' inválido. Escolha entre concat ou best.".format(mode=mode)
        )


GAP_TOKEN = "\uFFFF"


def _normalize_for_alignment(text: str) -> str:
    if not text:
        return ""

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("-\n", "")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized.strip()


def _tokenize(text: str) -> List[str]:
    return list(text)


def _align_tokens(
    anchor_tokens: List[str],
    candidate_tokens: List[str],
) -> List[Tuple[str, Optional[str], bool]]:
    matcher = difflib.SequenceMatcher(None, anchor_tokens, candidate_tokens, autojunk=False)
    alignment: List[Tuple[str, Optional[str], bool]] = []
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            for idx in range(a1 - a0):
                alignment.append((anchor_tokens[a0 + idx], candidate_tokens[b0 + idx], True))
        elif tag == "replace":
            len_a = a1 - a0
            len_b = b1 - b0
            length = max(len_a, len_b)
            for idx in range(length):
                if idx < len_a:
                    anchor_char = anchor_tokens[a0 + idx]
                    consumes_anchor = True
                else:
                    anchor_char = GAP_TOKEN
                    consumes_anchor = False
                candidate_char = candidate_tokens[b0 + idx] if idx < len_b else None
                alignment.append((anchor_char, candidate_char, consumes_anchor))
        elif tag == "delete":
            for idx in range(a1 - a0):
                alignment.append((anchor_tokens[a0 + idx], None, True))
        elif tag == "insert":
            for idx in range(b1 - b0):
                alignment.append((GAP_TOKEN, candidate_tokens[b0 + idx], False))
    return alignment


def _progressive_align(candidates: Dict[str, str], order: List[str]) -> Tuple[List[Dict[str, str]], str]:
    pivot_key = order[0]
    pivot_tokens = _tokenize(_normalize_for_alignment(candidates[pivot_key]))
    columns: List[Dict[str, str]] = [{pivot_key: ch} for ch in pivot_tokens]
    aligned_pivot = pivot_tokens[:]

    for key in order[1:]:
        candidate_tokens = _tokenize(_normalize_for_alignment(candidates[key]))
        aligned = _align_tokens(aligned_pivot, candidate_tokens)
        new_columns: List[Dict[str, str]] = []
        pivot_index = 0

        for anchor_char, candidate_char, consumes_anchor in aligned:
            if consumes_anchor:
                column = dict(columns[pivot_index])
                pivot_index += 1
            else:
                column = {pivot_key: ""}

            column[key] = candidate_char or ""
            new_columns.append(column)

        columns = new_columns
        aligned_pivot = [col.get(pivot_key, "") or GAP_TOKEN for col in columns]

    return columns, pivot_key


def _is_digit_like(ch: str) -> bool:
    return bool(ch) and ch.isdigit()


def _has_diacritic(ch: str) -> bool:
    if not ch:
        return False
    decomposed = unicodedata.normalize("NFD", ch)
    return any(unicodedata.category(c) == "Mn" for c in decomposed)


def _vote_column(column: Dict[str, str], engine_weights: Dict[str, float], pivot_key: str) -> str:
    scores: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    anchor_char = column.get(pivot_key, "")

    for key, char in column.items():
        if not char:
            continue
        weight = engine_weights.get(key, 1.0)
        if _is_digit_like(char):
            weight += 0.25
        if _has_diacritic(char):
            weight += 0.15
        scores[char] = scores.get(char, 0.0) + weight
        counts[char] = counts.get(char, 0) + 1

    if not scores:
        return ""

    best_score = max(scores.values())
    best_chars = [char for char, score in scores.items() if score == best_score]
    if len(best_chars) == 1:
        chosen = best_chars[0]
    elif anchor_char and anchor_char in best_chars:
        chosen = anchor_char
    else:
        chosen = sorted(best_chars)[0]

    if not anchor_char and chosen.isspace() and counts.get(chosen, 0) < 2:
        return ""

    return chosen


def _default_engine_weights(keys: List[str]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for key in keys:
        if key.startswith("paddle") or key == "paddle":
            weights[key] = 1.1
        elif key.startswith("easy") or key == "easy":
            weights[key] = 1.1
        else:
            weights[key] = 1.0
    return weights


def fuse_candidates(candidates: Dict[str, str], anchor_key: Optional[str] = None) -> str:
    filtered_candidates = {key: value for key, value in candidates.items() if value}
    if not filtered_candidates:
        return ""

    if anchor_key is None or anchor_key not in filtered_candidates:
        anchor_key = max(
            filtered_candidates.keys(),
            key=lambda key: len(_normalize_for_alignment(filtered_candidates[key])),
        )

    order = [anchor_key] + [key for key in sorted(filtered_candidates.keys()) if key != anchor_key]
    columns, pivot_key = _progressive_align(filtered_candidates, order)

    engine_weights = _default_engine_weights(list(filtered_candidates.keys()))
    fused_chars = [_vote_column(column, engine_weights, pivot_key) for column in columns]
    fused_text = "".join(fused_chars)
    fused_text = re.sub(r"[ \t]+", " ", fused_text)
    fused_text = re.sub(r" ?\n ?", "\n", fused_text)
    return unicodedata.normalize("NFC", fused_text.strip())

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
