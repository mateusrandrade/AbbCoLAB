
# Do Arquivo ao Algoritmo — OCR CLI (v0.3.0)

CLI colaborativa para **OCR multi-engine** (Tesseract + PaddleOCR + EasyOCR), curadoria humana (`*.curator.txt`) e exportação de dataset (`abbadia_train.jsonl`) para treinar o **AbbadiaT5** como pós-corretor/combinador.

## Instalação (CPU)
> Precisa do **binário do Tesseract** no sistema:
- Ubuntu/Debian: `sudo apt-get install -y tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: instalar via Chocolatey (`choco install tesseract`) ou MSI oficial.

Depois:
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\Activate.ps1)
pip install -e ".[all-cpu]"
```

## Uso rápido
### 1) OCR multi-engine (default: tesseract + paddle + easyocr)
```bash
daa ocr run   --input-dir data/colecao_01   --glob "**/*.jpg"   --lang por   --oem 3   --psm 3 4 6 11 12   --outputs txt
```

Gera por imagem:
```
pagina.tess.psm03.txt  pagina.tess.psm04.txt  pagina.tess.psm06.txt  pagina.tess.psm11.txt  pagina.tess.psm12.txt
pagina.paddle.txt      pagina.paddle.json
pagina.easy.txt        pagina.easy.json
```

### 2) Curadoria
Crie **um arquivo por imagem** com a verdade revisada:
```
pagina.curator.txt
```

### 3) Exportar dataset para o AbbadiaT5
```bash
daa export   --input-dir data/colecao_01   --glob "**/*.jpg"   --out data/colecao_01/exports/abbadia_train.jsonl
```
- 1 linha por página com `*.curator.txt` encontrado.
- `input_text` depende do `--multi-hyp`: `concat` (default) junta os candidatos com tags (`<tess psm=..>…</tess>`, `<paddle>…</paddle>`, `<easy>…</easy>`); `best` seleciona o candidato com menor CER em relação ao curator; `fuse` alinha as hipóteses e vota posição a posição para propor um único texto fundido.

| `--multi-hyp` | Estratégia | Quando usar |
|---------------|------------|-------------|
| `concat` | Concatena todas as hipóteses com tags de proveniência. | Ideal para treinar o AbbadiaT5 a combinar textos multi-engine. |
| `best` | Escolhe apenas o candidato com menor CER em relação ao curator. | Útil em coleções homogêneas com um motor dominante. |
| `fuse` | Alinha caractere a caractere e vota por coluna para gerar uma hipótese única já combinada. | Reduz o esforço de edição manual antes da curadoria colaborativa. |

- `target_text` = conteúdo do `*.curator.txt`.
- Manifest de export (`export_manifest.csv/jsonl`) indica o modo (`multi_hyp_mode`) e o(s) candidato(s) usados.
  - A partir da v0.3.0, os manifests possuem as colunas `multi_hyp_mode` e `selected_candidates` para preservar compatibilidade com análises anteriores.

### 4) Avaliação agregada
```bash
daa eval   --input-dir data/colecao_01   --glob "**/*.jpg"   --out-dir data/colecao_01/exports/eval
```
Saídas:
- `eval_by_page.csv` (CER/WER por candidato e página)
- `eval_summary_by_engine_psm.csv` (médias por engine/psm)

## GPU (opcional)
- **EasyOCR (PyTorch)**: instale `torch` compatível com sua CUDA (site do PyTorch).
- **PaddleOCR**: `paddlepaddle-gpu==<versão>` conforme sua CUDA (docs do Paddle). Use `daa ocr run --gpu true`.

## Docker/Devcontainer
Ambiente reprodutível:
```bash
docker build -t daa-ocr-cli:0.3.0 .
docker run --rm -it -v $PWD/data:/data daa-ocr-cli:0.3.0 daa version
```

Consulte `docs/Guia_de_Curadoria.md`.
