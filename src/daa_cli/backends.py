
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
from .utils import run_cmd, write_json

def run_tesseract(image: Path, lang: str, oem: int, psm_list: List[int], out_formats: List[str], dry_run: bool=False) -> List[Dict[str, Any]]:
    rows = []
    for psm in psm_list:
        out_base = image.with_suffix("").as_posix() + f".tess.psm{psm:02d}"
        for fmt in out_formats:
            cmd = ["tesseract", str(image), out_base, "-l", lang, "--oem", str(oem), "--psm", str(psm)]
            if fmt in {"tsv","hocr","pdf"}:
                cmd.append(fmt)
            if dry_run:
                rc, dt, err = 0, 0.0, "DRY-RUN"
            else:
                rc, dt, err = run_cmd(cmd)
            rows.append({
                "engine":"tesseract",
                "psm":psm,
                "format":fmt,
                "exit_code":rc,
                "duration_sec":dt,
                "stderr":err,
                "out_path": str(Path(out_base + f".{fmt}"))
            })
    return rows

def _safe_import(module: str):
    try:
        return __import__(module)
    except Exception as e:
        return None

_easyocr_cache: Dict[Tuple[Tuple[str, ...], bool], Any] = {}
_paddle_cache: Dict[Tuple[bool, str], Any] = {}


def clear_easyocr_cache() -> None:
    _easyocr_cache.clear()


def clear_paddle_cache() -> None:
    _paddle_cache.clear()


def clear_ocr_caches() -> None:
    clear_easyocr_cache()
    clear_paddle_cache()


def _get_easyocr_reader(langs: Tuple[str, ...], gpu: bool):
    easyocr = _safe_import("easyocr")
    if easyocr is None:
        return None
    key = (langs, bool(gpu))
    reader = _easyocr_cache.get(key)
    if reader is None:
        reader = easyocr.Reader(list(langs), gpu=bool(gpu))
        _easyocr_cache[key] = reader
    return reader


def _get_paddle_ocr(gpu: bool, lang: str):
    paddleocr = _safe_import("paddleocr")
    if paddleocr is None:
        return None
    key = (bool(gpu), lang)
    ocr = _paddle_cache.get(key)
    if ocr is None:
        ocr = paddleocr.PaddleOCR(use_angle_cls=True, use_gpu=bool(gpu), lang=lang)
        _paddle_cache[key] = ocr
    return ocr


def run_easyocr(image: Path, langs: List[str], gpu: bool=False) -> Dict[str, Any]:
    langs_key = tuple(langs)
    reader = _get_easyocr_reader(langs_key, gpu)
    if reader is None:
        return {"engine":"easyocr","available":False,"error":"easyocr não instalado"}
    result = reader.readtext(str(image), detail=1)  # [ [bbox, text, conf], ... ]
    words = []
    lines = []
    for item in result:
        bbox, text, conf = item[0], item[1], float(item[2])
        words.append({"bbox": bbox, "text": text, "conf": conf})
        lines.append(text)
    text_out = "\n".join(lines)
    txt_path = image.with_suffix(".easy.txt")
    json_path = image.with_suffix(".easy.json")
    txt_path.write_text(text_out, encoding="utf-8")
    write_json(json_path, {"engine":"easyocr","gpu":gpu,"words":words})
    return {"engine":"easyocr","available":True,"out_txt":str(txt_path),"out_json":str(json_path)}

def run_paddle(image: Path, gpu: bool=False) -> Dict[str, Any]:
    ocr = _get_paddle_ocr(gpu, "pt")
    if ocr is None:
        return {"engine":"paddle","available":False,"error":"paddleocr não instalado"}
    result = ocr.ocr(str(image), cls=True)
    words = []
    lines = []
    for page in result:
        for line in page:
            bbox = line[0]
            text = line[1][0]
            conf = float(line[1][1])
            words.append({"bbox": bbox, "text": text, "conf": conf})
            lines.append(text)
    text_out = "\n".join(lines)
    txt_path = image.with_suffix(".paddle.txt")
    json_path = image.with_suffix(".paddle.json")
    txt_path.write_text(text_out, encoding="utf-8")
    write_json(json_path, {"engine":"paddle","gpu":gpu,"words":words})
    return {"engine":"paddle","available":True,"out_txt":str(txt_path),"out_json":str(json_path)}
