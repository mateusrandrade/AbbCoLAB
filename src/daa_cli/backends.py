
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
import inspect
import os
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
_deepseek_cache: Dict[Tuple[Optional[str], str], Any] = {}


def clear_easyocr_cache() -> None:
    _easyocr_cache.clear()


def clear_paddle_cache() -> None:
    _paddle_cache.clear()


def clear_ocr_caches() -> None:
    clear_easyocr_cache()
    clear_paddle_cache()
    _deepseek_cache.clear()


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


def _select_deepseek_entrypoint(module: Any):
    for candidate in ("DeepSeekOCR", "DeepSeekOcr", "OCR"):
        if hasattr(module, candidate):
            return getattr(module, candidate)
    for candidate in ("load_model", "get_model", "build_model"):
        if hasattr(module, candidate):
            return getattr(module, candidate)
    return None


def _build_deepseek_instance(model_path: Optional[str], device: str):
    deepseek_ocr = _safe_import("deepseek_ocr")
    if deepseek_ocr is None:
        return None, "deepseek_ocr não instalado"
    entrypoint = _select_deepseek_entrypoint(deepseek_ocr)
    if entrypoint is None:
        return None, "deepseek_ocr sem entrypoint suportado"
    try:
        signature = inspect.signature(entrypoint)
    except (TypeError, ValueError):
        signature = None
    kwargs = {}
    if signature:
        if "model_path" in signature.parameters and model_path:
            kwargs["model_path"] = model_path
        if "checkpoint" in signature.parameters and model_path:
            kwargs["checkpoint"] = model_path
        if "device" in signature.parameters:
            kwargs["device"] = device
    try:
        instance = entrypoint(**kwargs) if callable(entrypoint) else entrypoint
    except Exception as exc:
        return None, f"falha ao inicializar DeepSeek-OCR: {exc}"
    return instance, ""


def _get_deepseek_ocr(model_path: Optional[str], gpu: bool):
    device = "cuda" if gpu else "cpu"
    key = (model_path, device)
    cached = _deepseek_cache.get(key)
    if cached is not None:
        return cached, ""
    instance, error = _build_deepseek_instance(model_path, device)
    if instance is None:
        return None, error
    _deepseek_cache[key] = instance
    return instance, ""


def _select_deepseek_infer(instance: Any) -> Optional[Callable[..., Any]]:
    for name in ("infer", "predict", "__call__", "ocr", "run"):
        if hasattr(instance, name):
            return getattr(instance, name)
    return None


def _normalize_deepseek_result(result: Any) -> Tuple[str, List[Dict[str, Any]]]:
    if isinstance(result, str):
        return result, []
    if isinstance(result, dict):
        text = result.get("text") or result.get("result") or ""
        lines = result.get("lines")
        if not text and isinstance(lines, list):
            text = "\n".join(str(line.get("text", "")) for line in lines)
        words = result.get("words") or result.get("tokens") or []
        return text, words if isinstance(words, list) else []
    if isinstance(result, list):
        lines = []
        words = []
        for item in result:
            if isinstance(item, dict):
                if "text" in item:
                    lines.append(str(item["text"]))
                words.append(item)
            else:
                lines.append(str(item))
        return "\n".join(lines), words
    return str(result), []


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


def run_deepseek(image: Path, gpu: bool=False) -> Dict[str, Any]:
    model_path = os.environ.get("DEEPSEEK_OCR_MODEL_PATH") or os.environ.get("DEEPSEEK_OCR_WEIGHTS")
    instance, error = _get_deepseek_ocr(model_path, gpu)
    if instance is None:
        return {"engine":"deepseek","available":False,"error":error or "deepseek_ocr não instalado"}
    infer_fn = _select_deepseek_infer(instance)
    if infer_fn is None:
        return {"engine":"deepseek","available":False,"error":"deepseek_ocr sem método de inferência compatível"}
    try:
        result = infer_fn(str(image))
    except Exception as exc:
        return {"engine":"deepseek","available":False,"error":f"falha na inferência DeepSeek-OCR: {exc}"}
    text_out, words = _normalize_deepseek_result(result)
    txt_path = image.with_suffix(".deepseek.txt")
    json_path = image.with_suffix(".deepseek.json")
    txt_path.write_text(text_out, encoding="utf-8")
    write_json(json_path, {"engine":"deepseek","gpu":gpu,"model_path":model_path,"words":words})
    return {"engine":"deepseek","available":True,"out_txt":str(txt_path),"out_json":str(json_path)}
