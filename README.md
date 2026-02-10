# AbbCoLAB — Laboratório Colaborativo *Do Arquivo ao Algoritmo*

> Do arquivo ao Algoritmo, da leitura à colaboração.

---

## Sobre o AbbCoLAB

O **AbbCoLAB** é um **laboratório colaborativo dedicado à curadoria digital de documentos históricos digitalizados**.  
Este repositório disponibiliza uma **pipeline operável via linha de comando (CLI)** que permite a parceiros do projeto executar, em suas próprias máquinas, o processo completo de **extração, transcrição, correção humana e exportação de dados em formato estruturado**.

Ao articular técnicas de **OCR**, **aprendizado de máquina** e **revisão colaborativa**, o AbbCoLAB busca **aprimorar a leitura e a correção automática de textos históricos**, fortalecendo o campo da **história digital** e **ampliando o acesso público a acervos documentais processados digitalmente**.

---

## Estrutura conceitual

O AbbCoLAB é parte integrante do ecossistema **Do Arquivo ao Algoritmo**, e conecta-se diretamente ao modelo de linguagem **AbbadiaT5**.  
Esses três componentes formam um ciclo contínuo de produção, aprendizado e aprimoramento coletivo:

| Componente | Descrição | Função principal |
|-------------|------------|------------------|
| **Do Arquivo ao Algoritmo** (`daa`) | Projeto guarda-chuva que integra pesquisa histórica, preservação digital, ciência de dados e aprendizado de máquina. | Cria as bases conceituais, metodológicas e técnicas do ecossistema. |
| **AbbCoLAB** | Laboratório colaborativo e pipeline CLI. | Coordena a produção distribuída de dados OCR e textos curados. |
| **AbbadiaT5** | Modelo de linguagem treinado com dados do AbbCoLAB. | Corrige automaticamente erros típicos de OCR em fontes históricas. |

### Fluxo geral da pipeline
1. **Extração** — leitura automática das imagens por múltiplos motores (Tesseract, PaddleOCR, EasyOCR, DeepSeek-OCR opcional).  
2. **Curadoria** — revisão humana e padronização das transcrições em arquivos `*.curator.txt`.  
3. **Exportação** — criação de conjuntos de dados (`abbadia_train.jsonl`) para treinar o modelo *AbbadiaT5*.  
4. **Avaliação** — cálculo de métricas de desempenho (CER/WER) e geração de manifestos de proveniência.

---

## Por que o nome `daa`?

O prefixo **`daa`** vem de **Do Arquivo ao Algoritmo**, e identifica toda a família de ferramentas desenvolvidas a partir do projeto.  
Cada execução da CLI — como:

```bash
daa ocr run
daa export
daa eval
```

simboliza o gesto que estrutura essa proposta: levar o arquivo ao algoritmo, documentando cada etapa da transformação dos acervos históricos em dados digitais.

## Colaboração e abertura

O AbbCoLAB é um espaço de ciência colaborativa e aberta.
Cada participante pode contribuir rodando a pipeline em seu próprio ambiente, revisando as transcrições automáticas e compartilhando os resultados curados.
Essas contribuições alimentam continuamente o modelo AbbadiaT5, aprimorando sua capacidade de leitura e correção automática.

> AbbCoLAB é uma iniciativa vinculada ao projeto “Do Arquivo ao Algoritmo”, dedicada à preservação, digitalização e interpretação automatizada de fontes históricas, com base em práticas colaborativas, éticas e reprodutíveis.

## Licença

Este repositório é distribuído sob a licença MIT.
Sinta-se livre para utilizar, modificar e compartilhar, citando o projeto *Do Arquivo ao Algoritmo*.
