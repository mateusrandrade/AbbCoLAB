
# Guia de Curadoria — Do Arquivo ao Algoritmo

Este guia padroniza a criação da **verdade de referência** por página (`*.curator.txt`).

## Princípios
1. **Uma página → um arquivo `*.curator.txt`** ao lado da imagem.
2. O `*.curator.txt` representa a **leitura natural** do texto, não a quebra do OCR.
3. **Preservar grafia histórica** quando ela for conteúdo; corrigir apenas **erros de OCR**.

## Regras
- **Quebras e hífens**: remova hífens de quebra de linha e **reconstrua** palavras e frases.
- **Parágrafos**: una linhas artificiais; mantenha a paragrafação original se identificável.
- **Pontuação e espaços**: normalize mínimos (espaços duplos, aspas inconsistentes).
- **Caracteres confusos**: corrija substituições típicas do OCR (1↔l, rn↔m, 0↔O), **sem modernizar a ortografia histórica**.
- **Ruídos**: remova “lixo” evidente (artefatos), mas preserve rubricamentos significativos.
- **Metadados**: não colocar no `*.curator.txt` (ficam no manifest/export).

## Exemplo
Entrada OCR (ruim):
```
O capitan- mor da villa
de S. Jose foi nome-
ado em 18–
```
Curadoria:
```
O capitão-mor da vila de São José foi nomeado em 18–
```

## Boas práticas
- Usar editor em **UTF-8**.
- Salvar **apenas texto puro** (`.txt`) sem formatação.
- Em dúvida estrutural, escolha a versão **mais legível** e registre nota editorial quando necessário.

## Fluxo de trabalho

### Fluxo de curadoria com hipótese fundida (`.fuse.txt`) → verdade revisada (`.curator.txt`)

**Objetivo:** acelerar a revisão humana sem abrir mão da diversidade de saídas das diferentes engines de OCR.

> A pipeline gera, ao lado de cada imagem, um arquivo **`pagina.fuse.txt`** contendo uma **hipótese combinada** (fusão) das leituras do Tesseract (vários PSM), PaddleOCR e EasyOCR.  
> Esse arquivo é um **rascunho unificado** para facilitar a edição. A **verdade de referência** continua sendo o arquivo **`pagina.curator.txt`**.

**Passo a passo**
1. **OCR**  
   `daa ocr run --input-dir <colecao> --glob "**/*.jpg"`  
   Saídas por página: `pagina.tess.psm03.txt`, `pagina.tess.psm04.txt`, `pagina.tess.psm06.txt`, `pagina.tess.psm11.txt`, `pagina.tess.psm12.txt`, `pagina.paddle.txt`, `pagina.easy.txt`.

2. **Hipótese fundida automática**
   Durante o `daa export`, a CLI gera `pagina.fuse.txt` **automaticamente** (por padrão), sem sobrescrever arquivos existentes.
   Use `daa export --no-write-hypothesis ...` para pular essa etapa ou `--hypothesis-suffix "<novo>.txt"` para personalizar o nome.

3. **Curadoria humana**  
   Edite `pagina.fuse.txt` e **salve como** `pagina.curator.txt`, aplicando as regras deste guia (reconstrução de palavras/frases, preservação de grafia histórica, normalização de mínimos).

4. **Exportação para treino**
   `daa export --input-dir <colecao> --glob "**/*.jpg" --out abbadia_train.jsonl`
   - Flags úteis: `--no-write-hypothesis` (desativa a geração do `.fuse.txt`) e `--hypothesis-suffix "<novo>.txt"`.
   O export usa:
   - `target_text` ← conteúdo do `*.curator.txt` (ground truth);  
   - `candidates` ← todas as hipóteses (sempre salvas no JSONL);  
   - `input_text`:
     - `concat` (padrão): concatena com tags de proveniência (ideal para treino combinador);  
     - `best`/`fuse`: o *data loader* reconstrói o `input_text` a partir de `candidates`, preservando a riqueza multi-engine no treino.
