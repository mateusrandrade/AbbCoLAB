
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

### Fluxo de curadoria com hipótese fundida (.fuse.txt) → verdade revisada (.curator.txt)
1. **Executar o OCR**: rode `daa ocr run` no conjunto desejado para gerar as leituras brutas por *engine* e combinação de PSM.
2. **Listar os candidatos**: confirme os arquivos que serão exportados — as saídas ficam organizadas por *engine*/PSM e servem como referência para checar leituras divergentes.
3. **Exportar a hipótese fundida**: utilize `daa export` no modo apropriado; além dos artefatos históricos, o comando passa a escrever o rascunho `<página>.fuse.txt` ao lado das imagens.
4. **Editar a hipótese**: abra o `<página>.fuse.txt`, ajuste manualmente a leitura com base nos candidatos disponíveis e aplique as regras deste guia.
5. **Salvar a verdade revisada**: quando a revisão estiver concluída, salve o resultado final como `<página>.curator.txt`. Apenas este arquivo representa a **verdade de referência**.

#### Campos preenchidos por `daa export`
- **Modo `concat`**: `candidates` guarda todas as leituras concatenadas; `target_text` replica a união crua e `input_text` preserva o fluxo original de entrada.
- **Modo `best`**: `candidates` indica a leitura escolhida automaticamente por *engine*/PSM; `target_text` traz a saída com maior confiança; `input_text` registra a combinação usada para comparação.
- **Modo `fuse`**: `candidates` agrega os trechos que formam a hipótese fundida; `target_text` corresponde ao conteúdo salvo em `<página>.fuse.txt` e serve de rascunho; `input_text` armazena o material que alimentou a fusão.

O arquivo `*.fuse.txt` é sempre um **ponto de partida**. Somente o `*.curator.txt`, revisado por humanos, deve ser tratado como *ground truth* e publicado.
