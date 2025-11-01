
from __future__ import annotations
from pydantic import BaseModel
from typing import List, Literal

OutputFmt = Literal["txt","tsv","hocr","pdf"]

class OCRConfig(BaseModel):
    input_dir: str
    glob: str = "**/*.jpg"
    lang: str = "por"
    oem: int = 3
    psm: List[int] = [3,4,6,11,12]
    outputs: List[OutputFmt] = ["txt"]
    write_manifest: bool = True
    dry_run: bool = False
    gold_suffix: str = ".curator.txt"
    engines: List[str] = ["tesseract","paddle","easyocr"]
    gpu: bool = False
    easyocr_langs: List[str] = ["pt"]

class ExportConfig(BaseModel):
    input_dir: str
    glob: str = "**/*.jpg"
    out: str
    gold_suffix: str = ".curator.txt"
    multi_hyp: str = "concat"
    fail_if_no_gold: bool = True
    write_hypothesis: bool = True
    hypothesis_suffix: str = ".fuse.txt"

class EvalConfig(BaseModel):
    input_dir: str
    glob: str = "**/*.jpg"
    gold_suffix: str = ".curator.txt"
    out_dir: str
