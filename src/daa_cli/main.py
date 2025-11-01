
from __future__ import annotations
import typer
from typing import List
from rich import print as rprint
from .config import OCRConfig, ExportConfig, EvalConfig
from .ocr import ocr_batch
from .export import export_dataset
from .eval import eval_collection

app = typer.Typer(help="Do Arquivo ao Algoritmo — OCR CLI")

@app.command("version")
def version():
    rprint("[bold]daa-ocr-cli[/bold] v0.3.0")

ocr_app = typer.Typer(help="Comandos de OCR")
app.add_typer(ocr_app, name="ocr")

@ocr_app.command("run")
def ocr_run(
    input_dir: str = typer.Option(..., help="Diretório de entrada"),
    glob: str = typer.Option("**/*.jpg", help="Padrão glob"),
    lang: str = typer.Option("por", help="Idioma Tesseract"),
    oem: int = typer.Option(3, help="OEM (0-3)"),
    psm: List[int] = typer.Option([3,4,6,11,12], help="Lista de PSMs (Tesseract)"),
    outputs: List[str] = typer.Option(["txt"], help="txt/tsv/hocr/pdf"),
    write_manifest: bool = typer.Option(True, help="Grava manifest CSV/JSONL"),
    dry_run: bool = typer.Option(False, help="Apenas simula"),
    engines: List[str] = typer.Option(["tesseract","paddle","easyocr"], help="tesseract paddle easyocr"),
    gpu: bool = typer.Option(False, help="Usar GPU (Paddle/EasyOCR)"),
    easyocr_langs: List[str] = typer.Option(["pt"], help="Idiomas EasyOCR, ex.: pt en"),
):
    cfg = OCRConfig(
        input_dir=input_dir, glob=glob, lang=lang, oem=oem,
        psm=list(psm), outputs=list(outputs),
        write_manifest=write_manifest, dry_run=dry_run,
        engines=list(engines), gpu=gpu, easyocr_langs=list(easyocr_langs)
    )
    res = ocr_batch(cfg)
    rprint(res)

@app.command("export")
def export_cmd(
    input_dir: str = typer.Option(..., help="Diretório base a varrer"),
    glob: str = typer.Option("**/*.jpg", help="Arquivos de imagem base"),
    out: str = typer.Option(..., help="Arquivo JSONL de saída (dataset AbbadiaT5)"),
    gold_suffix: str = typer.Option(".curator.txt", help="Sufixo dos textos revisados"),
    multi_hyp: str = typer.Option(
        "concat",
        help="Como combinar hipóteses: concat (tags), best (melhor CER) ou fuse (alinhamento + votação)",
    ),
    fail_if_no_gold: bool = typer.Option(True, help="Falhar se nenhum curator for encontrado"),
):
    cfg = ExportConfig(
        input_dir=input_dir, glob=glob, out=out,
        gold_suffix=gold_suffix, multi_hyp=multi_hyp, fail_if_no_gold=fail_if_no_gold
    )
    res = export_dataset(cfg)
    rprint(res)

@app.command("eval")
def eval_cmd(
    input_dir: str = typer.Option(..., help="Diretório base a varrer"),
    glob: str = typer.Option("**/*.jpg", help="Arquivos de imagem base"),
    gold_suffix: str = typer.Option(".curator.txt", help="Sufixo dos textos revisados"),
    out_dir: str = typer.Option(..., help="Diretório de saída dos relatórios"),
):
    cfg = EvalConfig(input_dir=input_dir, glob=glob, gold_suffix=gold_suffix, out_dir=out_dir)
    res = eval_collection(cfg)
    rprint(res)

if __name__ == "__main__":
    app()
