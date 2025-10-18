
from __future__ import annotations
import subprocess as sp, time, json, csv
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional, Tuple

IMAGE_EXTS = {".jpg",".jpeg",".png",".tif",".tiff",".bmp",".pbm",".pgm",".ppm",".webp"}

def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def tesseract_version() -> str:
    try:
        out = sp.check_output(["tesseract", "--version"], text=True, stderr=sp.STDOUT).splitlines()[0]
        return out.strip()
    except Exception as e:
        return f"unknown ({e})"

def run_cmd(cmd: List[str]) -> Tuple[int, float, str]:
    t0 = time.time()
    try:
        cp = sp.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        stderr = (cp.stderr or "").strip()
        return cp.returncode, dt, stderr
    except Exception as e:
        return 1, time.time()-t0, str(e)

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_csv(path: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    ensure_parent(path)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)

def discover_images(input_dir: Path, glob: str) -> List[Path]:
    return [p for p in input_dir.glob(glob) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

def base_for_image(img: Path) -> Path:
    return img.with_suffix("")

def read_text_if_exists(p: Path) -> Optional[str]:
    if p.exists():
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return p.read_text(errors="ignore")
    return None

def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
