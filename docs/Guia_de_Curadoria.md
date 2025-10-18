
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
