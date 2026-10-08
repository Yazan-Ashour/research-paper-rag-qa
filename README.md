# Research Paper Q&A: A Retrieval-Augmented Generation System

A retrieval-augmented generation (RAG) system that answers questions about NLP research papers. It retrieves relevant passages with a hybrid dense and keyword search, generates a grounded answer with a locally hosted language model, and shows the supporting sentences in the sources. The project includes a Streamlit interface, a feedback system, and a set of evaluations that justify each design choice.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Dataset](#dataset)
5. [Tools and Models](#tools-and-models)
6. [Project Structure](#project-structure)
7. [Setup](#setup)
8. [Usage](#usage)
9. [Sample Inputs and Outputs](#sample-inputs-and-outputs)
10. [Evaluation and Model Selection](#evaluation-and-model-selection)
11. [Web Interface](#web-interface)
12. [Bonus Tasks](#bonus-tasks)
13. [Limitations](#limitations)
14. [Reproducing the Results](#reproducing-the-results)
15. [Acknowledgements](#acknowledgements)

## Overview

The system answers a user question in four steps: it embeds the question, retrieves the top-k most relevant text chunks from a vector store, passes those chunks and the question to a language model, and returns the answer together with its sources. If the documents do not contain the answer, the model is instructed to say so instead of guessing.

The corpus is a subset of the QASPER dataset, which provides human-written questions with gold answers and gold evidence paragraphs. That ground truth lets the project measure retrieval quality and compare models, instead of judging answers by eye.

## Features

- Section-aware text cleaning and chunking (at most 250 words per chunk, sentence aligned)
- Hybrid retrieval that combines dense embeddings (FAISS) with BM25 keyword scores
- Optional restriction of the search to a single paper
- Grounded answer generation with numbered citations and an explicit refusal rule
- Evaluation of retrieval quality against QASPER gold evidence
- Systematic comparisons of embedding models, retrieval settings and language models, each saved as a JSON report
- Streamlit interface with source highlighting and thumbs up/down feedback

## Architecture

```
                  Offline (build the index)
  QASPER papers -> clean text -> chunk (<=250 words) -> embed -> FAISS index
                                                   \-> BM25 index

                  Online (answer a question)
  Question -> embed -> dense scores (FAISS) --\
           -> tokenize -> BM25 scores ---------> hybrid score (0.5 / 0.5) -> top-k chunks
                                                                                 |
  Answer + sources + highlights  <-  language model  <-  prompt (context + question)
```

Hybrid score = `dense_weight * dense + (1 - dense_weight) * bm25`, with both score sets min-max normalized per query. The default `dense_weight` is 0.5.

## Dataset

The project uses [QASPER](https://huggingface.co/datasets/allenai/qasper), a dataset of questions about NLP research papers, written by readers who had seen only the abstract. It is released under CC BY 4.0.

| Property | Value |
|---|---|
| Split used | `validation` |
| Selection | First 8 papers that match topic keywords (retrieval, question answering, language model, reading comprehension, information extraction) and have at least 4 questions |
| Chunks | 211 chunks, about 140 words on average |
| Questions | 33 (32 with gold evidence, 1 unanswerable) |

Example papers in the corpus include *Improved Neural Relation Detection for Knowledge Base Question Answering*, *Sensitive Data Detection and Classification in Spanish Clinical Text: Experiments with BERT*, *A Framework for Evaluation of Machine Reading Comprehension Gold Standards*, *Dice Loss for Data-imbalanced NLP Tasks*, *VAIS ASR: Building a conversational speech recognition system using language model combination* and *What Does My QA Model Know? Devising Controlled Probes using Expert Knowledge*.

**Preprocessing.** QASPER text contains reference tokens such as `BIBREF12` and `FIGREF3`. The cleaning step removes them together with the empty brackets and repeated punctuation they leave behind. Each paper is split by section, abstract first, and chunks never cross a section boundary. Tables and figures are not part of the QASPER text, so evidence that points to them is excluded from the evaluation.

## Tools and Models

| Component | Choice | Notes |
|---|---|---|
| Language | Python 3.10 to 3.12 | |
| Dataset loading | Hugging Face `datasets` | Loaded from the `refs/convert/parquet` revision |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | 384 dimensions, runs locally. Selected by comparison |
| Vector store | FAISS `IndexFlatIP` | Exact cosine search on normalized vectors |
| Keyword search | `rank-bm25` (BM25Okapi) | Blended with the dense scores |
| Language model | `llama3.2:3b` through Ollama | Temperature 0.1, 400 max tokens. Selected by comparison |
| LLM client | `openai` Python package | Works with any OpenAI-compatible endpoint |
| Interface | Streamlit | |

The client reads its endpoint from `.env`, so a different provider only needs three changed values. NVIDIA NIM was tried first, but its endpoints returned 404 errors for the account used, so a local Ollama model was used instead.

## Project Structure

```
rag_project/
├── data_prep.py            Load, clean and chunk QASPER; build test questions
├── vector_store.py         Embeddings, FAISS index, BM25, hybrid search, persistence
├── rag_pipeline.py         Retrieval, prompt, generation, command-line demo
├── evaluate.py             Retrieval hit rate against gold evidence
├── compare_embeddings.py   Embedding model comparison
├── grid_search.py          Chunk size, dense weight and top-k search
├── compare_llms.py         Language model comparison
├── highlighting.py         Supporting-sentence highlighting
├── feedback.py             JSON Lines feedback store
├── app.py                  Streamlit interface
├── reporting.py            Shared JSON report writer
├── requirements.txt
├── .env.example
├── README.md
├── screenshots/                   Interface screenshots
├── results/                JSON evidence reports (created by the scripts)
├── index/                  FAISS index and chunk metadata (created by vector_store.py)
└── feedback/               User feedback log (created by the interface)
```

## Setup

### Prerequisites

- Python 3.10 to 3.12
- [Ollama](https://ollama.com) installed and running
- About 2 GB of free disk space for the language model, plus space for PyTorch

### Installation

```bash
git clone <https://github.com/Yazan-Ashour/research-paper-rag-qa>
cd rag_project

python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env`:

```text
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2:3b
```

Ollama ignores the API key, but the client requires a non-empty value. To use another OpenAI-compatible provider, change these three values.

### Language model

```bash
ollama pull llama3.2:3b
ollama list
```

### Build the index

```bash
python vector_store.py
```

The first run downloads the dataset and the embedding model, builds the chunks, indexes them, and saves the result to `index/`.

## Usage

**Command-line demo.** Runs five paper-scoped QASPER questions and one out-of-scope question:

```bash
python rag_pipeline.py
```

**Web interface:**

```bash
streamlit run app.py
```

It opens at `http://localhost:8501`.

**Use the pipeline from Python:**

```python
from rag_pipeline import RAGPipeline
from vector_store import VectorStore

pipeline = RAGPipeline(VectorStore.load())
result = pipeline.ask("What does KBQA stand for?")
print(result.answer)
for chunk, score in result.sources:
    print(score, chunk.title, chunk.section)
```

Pass `paper_id=...` to `ask` to restrict the search to one paper.

## Sample Inputs and Outputs

Outputs below come from the final configuration (`llama3.2:3b`, hybrid retrieval, top-k 5). The first six use paper-scoped retrieval, as in `rag_pipeline.py`. Gold answers are taken from QASPER, and the one for the framework question is abridged.

| # | Question | System answer | Gold answer |
|---|---|---|---|
| 1 | On which benchmarks they achieve the state of the art? | They achieve the state-of-the-art on both SimpleQuestions and WebQSP. [2] (1st row of results), [3] (last row of Table 2). | SimpleQuestions, WebQSP |
| 2 | What they use in their propsoed framework? | They use two types of relation representation: word-level and relation-level. The word-level focuses on local information (words and short phrases), while the relation-level focuses on global information (long phrases and skip-grams). [3] | Relation-level and word-level representations, BiLSTMs, residual learning (abridged) |
| 3 | What does KBQA abbreviate for | KBQA stands for Knowledge Base Question Answering. | Knowledge Base Question Answering |
| 4 | What is te core component for KBQA? | Relation detection is a core component for Knowledge Base Question Answering (KBQA). [1] | Annotators gave loosely worded answers; the answer is plausible |
| 5 | What is the performance of BERT on the task? | BERT achieves a remarkably high recall of 0.979 in the detection scenario, improving by more than 4 points the second-best system, spaCy. [1] [4] | F1 scores per dataset and task (reported in tables) |
| 6 | Who won the 2018 FIFA World Cup? (out of scope) | I could not find the answer in the provided documents. | None |

Examples from the web interface, searching all papers:

| Question | System answer |
|---|---|
| Which clinical datasets were used in the paper on sensitive data detection with BERT? | The two clinical datasets used in the paper are MEDDOCAN: Medical Document Anonymization shared task dataset and NUBes, a corpus of real medical reports in Spanish. [1] [4] |
| Which datasets are used for the machine reading comprehension experiments? | The SQuAD v1.1, SQuAD v2.0, and Quoref datasets are used for the machine reading comprehension experiments [4]. |
| What does the DictionaryQA dataset is created from | The DictionaryQA dataset is created from the GCIDE dictionary, which is a comprehensive open-source English dictionary built largely from the Webster's Revised Unabridged Dictionary [1]. |
| What accuracy does GPT-4 achieve on SQuAD? (out of scope) | I could not find the answer in the provided documents. |

## Evaluation and Model Selection

Every design choice below was compared on the same QASPER questions. Each comparison writes a JSON report to `results/`, so the decisions can be checked.

### Retrieval metric

A question counts as a hit at k when one of the top-k retrieved chunks covers at least half of a gold evidence paragraph, measured by 4-word sequence overlap. Evidence that points to tables or figures is excluded. 32 questions have gold evidence, so one question is worth about 3.1 percentage points.

Two modes are reported. **Paper-scoped** restricts the search to the paper the question was written for, which matches how QASPER was built. **Global** searches all papers, which is harder because many QASPER questions only make sense inside their own paper.

### Final retrieval results

Report: `results/retrieval_evaluation.json`

| Mode | Hit@1 | Hit@3 | Hit@5 |
|---|---|---|---|
| Paper-scoped | 31.2% (10/32) | 65.6% (21/32) | 81.2% (26/32) |
| Global | 18.8% (6/32) | 43.8% (14/32) | 59.4% (19/32) |

### Embedding model comparison

Report: `results/embedding_comparison.json`. Run with dense-only retrieval, before hybrid search was added.

| Model | Scoped Hit@1 / 3 / 5 | Global Hit@1 / 3 / 5 | Overall | Index build |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** | 25.0 / 65.6 / 81.2 | 15.6 / 40.6 / 46.9 | **45.8%** | 5.3 s |
| bge-small-en-v1.5 | 31.2 / 59.4 / 78.1 | 18.8 / 34.4 / 40.6 | 43.8% | 16.8 s |
| multi-qa-MiniLM-L6-cos-v1 | 21.9 / 53.1 / 75.0 | 15.6 / 25.0 / 40.6 | 38.5% | 8.5 s |

The overall score is the mean of all six hit rates. `all-MiniLM-L6-v2` was selected because it had the best Hit@3 and Hit@5 in both modes and the fastest indexing. BGE led only at Hit@1, by 1 to 2 questions, and the gap between the top two models is within noise on 32 questions.

### Retrieval settings grid search

Report: `results/grid_search.json`. 48 combinations of chunk size (100, 150, 200, 250 words), dense weight (0.3, 0.5, 0.7, 1.0) and top-k (3, 5, 8). A question counts as a hit when the retrieved chunks together cover the evidence.

| Setting | Average score by value |
|---|---|
| Chunk size | 100: 53.3%, 150: 54.6%, 200: 60.7%, **250: 63.4%** |
| Dense weight | 0.3: 56.5%, **0.5: 60.2%**, 0.7: 59.5%, 1.0 (dense only): 55.7% |
| Top-k | 3: 45.8%, 5: 60.2%, **8: 68.0%** |

| Configuration (chunk / dense weight / top-k) | Scoped | Global | Score | Context sent to the LLM |
|---|---|---|---|---|
| 250 / 1.0 / 5 (baseline, dense only) | 81.2% | 46.9% | 64.1% | 712 words |
| **250 / 0.5 / 5 (selected)** | 81.2% | **59.4%** | **70.3%** | 728 words |
| 200 / 0.5 / 8 (highest score) | 87.5% | 62.5% | 75.0% | 1006 words |

The selected configuration raises global Hit@5 by 4 questions at the same context size. The highest-scoring configuration sends about 38% more text to a 3B model for a gain of 1 to 2 questions, so it was not adopted.

### Language model comparison

Report: `results/llm_comparison.json`. 20 in-scope questions sampled with a fixed seed, plus 5 out-of-scope questions that should be refused. Every model receives identical retrieved context.

Metric definitions: **F1** and **recall** are token overlap with the best matching gold answer. **Cited** is the share of answers with a `[n]` citation. **False refusal** is the share of answerable questions the model refused. **Correct refusal** is the share of out-of-scope questions it refused.

**Final configuration** (top-k 5, hybrid retrieval, final prompt):

| Model | F1 | Recall | Cited | False refusal | Correct refusal | Sec/answer |
|---|---|---|---|---|---|---|
| gemma3:1b | 0.218 | 0.409 | 25% | 0% | 20% | 0.8 |
| **llama3.2:3b** | 0.218 | **0.522** | 75% | **5%** | 100% | 1.6 |
| qwen2.5:3b | 0.216 | 0.467 | 75% | 20% | 100% | 1.4 |
| qwen3:8b | 0.171 | 0.336 | 60% | 10% | 100% | 39.3 |

**Baseline** (top-k 4, dense only, first prompt), kept in `results/llm_comparison_baseline.json`:

| Model | F1 | Recall | Cited | False refusal | Correct refusal | Sec/answer |
|---|---|---|---|---|---|---|
| gemma3:1b | 0.179 | 0.240 | 5% | 15% | 60% | 0.6 |
| llama3.2:3b | 0.227 | 0.407 | 70% | 35% | 100% | 1.2 |
| qwen2.5:3b | 0.233 | 0.385 | 35% | 45% | 100% | 1.1 |
| qwen3:8b | 0.160 | 0.371 | 45% | 20% | 100% | 41.4 |

`llama3.2:3b` was selected. F1 is a three-way tie, so the decision rests on the other metrics: it has the highest recall, a 5% false-refusal rate, a 75% citation rate and perfect refusal of out-of-scope questions. `gemma3:1b` answered 4 of 5 out-of-scope questions from its own knowledge. `qwen2.5:3b` refused 20% of answerable questions. `qwen3:8b` is a reasoning model that took about 39 seconds per answer and returned 4 empty answers because hidden reasoning consumed the token limit.

Between the two runs, `llama3.2:3b` false refusals fell from 35% to 5% and recall rose from 0.407 to 0.522. Those gains come from the combined effect of top-k 5, hybrid retrieval, a revised prompt, and removing passage titles from the model context, which stopped the model from echoing them in answers.

## Web Interface

The Streamlit app answers questions in a chat layout. The sidebar selects a single paper or all papers, sets the number of retrieved passages, offers example questions, and shows the running feedback count.

![Answer with feedback controls](docs/ui_answer_feedback.png)

Cited sources open automatically, with the sentences that support the answer highlighted. Uncited sources stay collapsed.

![Highlighted source passage](docs/ui_highlighted_sources.png)

![DictionaryQA example](docs/ui_dictionaryqa.png)

Searching a single paper:

![Paper-scoped search](docs/ui_scoped_search.png)

## Bonus Tasks

| Task | Status | Implementation |
|---|---|---|
| Feedback system | Done | Thumbs up/down with an optional comment, saved to `feedback/feedback.jsonl` with the question, answer, model and source list |
| Highlight exact text sources | Done | `highlighting.py` marks the sentences in cited passages that share the most content words with the answer |
| Simple UI | Done | Streamlit app in `app.py` |
| Arabic or multilingual support | Planned | Not implemented yet |

## Limitations

- **Small evaluation set.** 32 questions with gold evidence and 20 for the language model comparison. One question is worth 3 to 5 points, so small gaps are noise.
- **No held-out test set.** The embedding model, retrieval settings and language model were all chosen on the same questions, so the reported scores are optimistically biased, particularly the best row of the grid search.
- **Gold evidence from tables is missing.** QASPER text excludes tables and figures, so numeric questions such as the BERT performance example cannot be fully answered or evaluated.
- **Approximate hit metric.** Coverage by 4-gram overlap stands in for exact paragraph matching, because chunks can merge or split the original paragraphs.
- **Small-model weaknesses.** The 3B model sometimes invents small details (for example, "1st row of results" in sample 1), places citations inconsistently, and can refuse questions it could answer.
- **Refusals rely on the model.** Hybrid scores are min-max normalized per query, so the top chunk always scores near 1, even for irrelevant questions. A similarity threshold cannot detect out-of-scope questions, and the out-of-scope test has only 5 questions.
- **Heuristic highlighting.** It scores sentences by word overlap and favors short, dense sentences. Long supporting sentences can be missed, and paraphrased answers may highlight nothing. It is not model-based attribution.
- **Ambiguous questions.** Many QASPER questions depend on knowing the paper. Global search is much weaker than paper-scoped search (59.4% against 81.2% Hit@5).
- **Verbose answers under token F1.** Gold answers are short spans and the model writes sentences, so recall is a fairer metric than F1 here.
- **English only.** The embedding model and corpus are English.
- **Evaluation reports reflect different stages.** The embedding comparison was run with dense-only retrieval. Reproducing it with the current code needs `DENSE_WEIGHT = 1.0`.

## Reproducing the Results

Run the scripts from the activated environment, after `ollama pull llama3.2:3b` and `python vector_store.py`.

```bash
python evaluate.py            # results/retrieval_evaluation.json
python compare_embeddings.py  # results/embedding_comparison.json (set DENSE_WEIGHT = 1.0 in vector_store.py to match the reported run)
python grid_search.py         # results/grid_search.json
python compare_llms.py        # results/llm_comparison.json (pull each candidate model first)
```

The language model comparison needs `gemma3:1b`, `llama3.2:3b`, `qwen2.5:3b` and `qwen3:8b` installed in Ollama, and skips any that are missing.

## Acknowledgements

- Dataset: Dasigi et al. (2021), *A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers*, NAACL 2021. Licensed CC BY 4.0.
- Embedding model: [sentence-transformers](https://www.sbert.net/) `all-MiniLM-L6-v2`.
- Vector search: [FAISS](https://github.com/facebookresearch/faiss). Keyword search: [rank-bm25](https://github.com/dorianbrown/rank_bm25).
- Language models served locally with [Ollama](https://ollama.com).