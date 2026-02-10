# Como usar o AbbCoLAB CLI (daa-ocr-cli)

Guia completo para instalar, configurar e utilizar a CLI do **AbbCoLAB** em ambiente Windows, macOS ou Linux, mesmo que você **nunca tenha usado Python antes**.

---

## O que é o AbbCoLAB CLI?
A `daa-ocr-cli` (v0.3.2) é a interface em linha de comando do AbbCoLAB para rodar OCR multi-engine (Tesseract, PaddleOCR, EasyOCR e DeepSeek-OCR), registrar curadoria humana (`*.curator.txt`), exportar datasets JSONL para treinar o modelo AbbadiaT5 e calcular métricas agregadas (CER/WER).

---

## Requisitos básicos
| Componente | Função | Onde obter |
|------------|--------|-----------|
| **Python 3.9+** | Necessário para rodar a CLI | [python.org/downloads](https://www.python.org/downloads/) |
| **Tesseract** (obrigatório) | Engine principal de OCR | `apt-get install tesseract-ocr` (Ubuntu/Debian); `brew install tesseract` (macOS); `choco install tesseract` ou MSI (Windows) |
| **PaddleOCR / EasyOCR (opcionais)** | Engines adicionais de OCR | Instaladas via `pip` pelo extra `all-cpu`; GPU requer builds específicos |
| **DeepSeek-OCR (opcional)** | Engine OCR via modelo DeepSeek | Repositório DeepSeek-OCR instalado localmente + variável de ambiente com pesos |
| **(Opcional) GPU** | Aceleração para PaddleOCR/EasyOCR | Instale `torch` ou `paddlepaddle-gpu` compatíveis com sua CUDA |

> **CPU x GPU x CUDA (em linguagem simples)**
> - **CPU:** é o processador central (faz tudo, mas com menos velocidade para tarefas grandes de IA).
> - **GPU:** é a placa de vídeo. Ela tem muitos núcleos e acelera redes neurais (OCR mais rápido).
> - **CUDA:** é o "driver de aceleração" da NVIDIA que permite a programas (como PyTorch/Paddle) usarem a GPU. Só funciona em GPUs NVIDIA e requer drivers atualizados.

> A CLI já prepara saídas e manifests mesmo sem GPU; o uso de GPU é apenas para acelerar PaddleOCR, EasyOCR e DeepSeek-OCR.

---

## Etapa 1 – Preparar o ambiente Python (passo a passo para iniciantes)

### 1.1 Instalar o Python
- **Windows (recomendado via instalador oficial):**
  1. Acesse [python.org/downloads](https://www.python.org/downloads/) e clique em **Download Python 3.x.x**.
  2. Abra o instalador e marque a opção **Add Python to PATH** antes de clicar em **Install Now**.
  3. Quando terminar, abra o **PowerShell** e confirme a instalação:
     ```powershell
     python --version
     pip --version
     ```
     Você deve ver algo como `Python 3.11.x` e `pip 24.x.x`.

- **macOS (Homebrew):**
  1. Abra o **Terminal**.
  2. Instale o Homebrew (se ainda não tiver) conforme <https://brew.sh>.
  3. Instale o Python e confirme:
     ```bash
     brew install python
     python3 --version
     python3 -m pip --version
     ```

- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt update
  sudo apt install -y python3 python3-venv python3-pip
  python3 --version
  python3 -m pip --version
  ```

> Dica: Se já tiver Python instalado, apenas confirme as versões. A CLI funciona a partir do Python 3.9.

### 1.2 Criar e ativar um ambiente virtual (para isolar as dependências)
1. No terminal/powershell, navegue até a pasta onde o projeto está salvo (ex.: `cd C:\\Projetos\\abbcolab` ou `cd ~/Projetos/abbcolab`).
2. Crie o ambiente virtual:
   - Windows: `python -m venv .venv`
   - macOS/Linux: `python3 -m venv .venv`
3. Ative o ambiente virtual:
   - Windows (PowerShell): `.\\.venv\\Scripts\\Activate.ps1`
   - macOS/Linux: `source .venv/bin/activate`
4. Atualize o `pip` dentro do ambiente virtual:
   ```bash
   python -m pip install --upgrade pip
   ```

Você saberá que o ambiente está ativo ao ver `(.venv)` no início da linha do terminal.

### 1.3 Instalar dependências da CLI
Com o ambiente virtual ativo, instale a CLI e os pacotes de CPU:
```bash
pip install -e '.[all-cpu]'
```

- Esse comando instala a própria CLI (`daa`) e as dependências de OCR em modo CPU.
- **Se for usar GPU (PaddleOCR/EasyOCR):** antes do comando acima, instale **uma** biblioteca de GPU compatível:
  - **PyTorch** (para EasyOCR): escolha a versão conforme sua CUDA em <https://pytorch.org/get-started/locally/>. Exemplo (CUDA 12.1):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
  - **PaddlePaddle GPU** (para PaddleOCR): escolha a versão conforme sua CUDA em <https://www.paddlepaddle.org.cn/>. Exemplo (Linux, CUDA 11.x):
    ```bash
    pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/gpu
    ```
  - **Como saber sua CUDA/GPU?** Se tiver driver NVIDIA instalado, rode `nvidia-smi` no terminal (Linux/Windows). Se não houver GPU NVIDIA, instale apenas em CPU.
  - Depois de instalar `torch` **ou** `paddlepaddle-gpu`, finalize com:
    ```bash
    pip install -e '.[all-cpu]'
    ```
    para registrar a CLI e demais dependências no mesmo ambiente virtual.

---

## Etapa 2 – Executar OCR multi-engine
Com o venv ativo, rode:
```bash
daa ocr run \
  --input-dir data/colecao_01 \
  --glob "**/*.jpg" \
  --lang por \
  --oem 3 \
  --psm 3 4 6 11 12 \
  --outputs txt \
  --engines tesseract paddle easyocr \
  --gpu false
```

- Saídas por imagem: `*.tess.psmXX.txt`, `*.paddle.txt/.json`, `*.easy.txt/.json`, `*.deepseek.txt/.json` (quando ativado).
- A CLI grava manifestos CSV/JSONL em `manifests/ocr_manifest.*` por padrão.

### DeepSeek-OCR (opcional)
O backend DeepSeek-OCR usa o módulo Python **`deepseek_ocr`** e instancia a classe **`DeepSeekOCR`** (ponto de entrada oficial), chamando o método de inferência `infer(...)` para gerar o texto. Para habilitar:

1. Clone/instale o repositório **DeepSeek-OCR** no mesmo ambiente virtual.
2. Disponibilize o caminho dos pesos localmente com uma das variáveis:
   - `DEEPSEEK_OCR_MODEL_PATH=/caminho/para/pesos-ou-modelo`
   - `DEEPSEEK_OCR_WEIGHTS=/caminho/para/pesos-ou-modelo`
3. Rode a CLI com `--engines deepseek` (ou combinado com outros engines).

Exemplo:
```bash
export DEEPSEEK_OCR_MODEL_PATH="$HOME/models/deepseek-ocr"
daa ocr run \
  --input-dir data/colecao_01 \
  --glob "**/*.jpg" \
  --engines deepseek \
  --gpu false
```

---

## Etapa 3 – Fazer a curadoria manual
Revise cada página criando um arquivo `*.curator.txt` na mesma pasta da imagem (um por página). Ex.: `pagina.curator.txt` contendo a transcrição corrigida.

---

## Etapa 4 – Exportar dataset para o AbbadiaT5
Gere o JSONL consolidado (uma linha por página com curator encontrado):
```bash
daa export \
  --input-dir data/colecao_01 \
  --glob "**/*.jpg" \
  --out data/colecao_01/exports/abbadia_train.jsonl \
  --multi-hyp concat \
  --gold-suffix .curator.txt
```

- `concat` (padrão) junta candidatos com tags; `best` pega o menor CER; `fuse` alinha e vota caractere a caractere.
- Por padrão, também grava `pagina.fuse.txt`; desative com `--no-write-hypothesis` ou mude o sufixo com `--hypothesis-suffix`.
- O manifest (`export_manifest.csv/jsonl`) traz `multi_hyp_mode` e `selected_candidates` para auditoria.

---

## Etapa 5 – Avaliar a coleção
Calcule CER/WER por página e resumos por engine/PSM:
```bash
daa eval \
  --input-dir data/colecao_01 \
  --glob "**/*.jpg" \
  --gold-suffix .curator.txt \
  --out-dir data/colecao_01/exports/eval
```

Saídas esperadas:
- `eval_by_page.csv` (CER/WER por candidato e página)
- `eval_summary_by_engine_psm.csv` (médias por engine/psm)

---

## Dicas rápidas
- Use `daa version` para conferir a versão da CLI (v0.3.2).
- `--gpu true` acelera PaddleOCR/EasyOCR se você tiver CUDA instalada; Tesseract roda sempre em CPU.
- Ajuste `--outputs` para `txt`, `tsv`, `hocr` ou `pdf` conforme a necessidade.
- Garanta que o Tesseract esteja no PATH antes de iniciar o OCR; é o único requisito obrigatório listado na instalação.

---

Pronto! Agora você pode rodar o AbbCoLAB CLI, curar as transcrições e exportar datasets para treinar o AbbadiaT5.
