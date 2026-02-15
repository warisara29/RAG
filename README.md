# Thai Criminal Law Q&A Chatbot: PageIndex vs PageIndex + Light Knowledge Graph

**Course:** CS652 Applied Machine Learning
**Student:** 6809036087
**Due:** 14 February 2026

---

## Overview

This project compares two retrieval-augmented generation (RAG) approaches for a Thai Criminal Law Q&A Chatbot:

1. **Pipeline A: PageIndex** — Vector-free RAG using a hierarchical tree index with LLM-based navigation
2. **Pipeline B: PageIndex + Light Knowledge Graph** — Same tree index augmented with cross-reference graph expansion

Both pipelines use the **Thai Criminal Code** (ประมวลกฎหมายอาญา, 444 sections) as the knowledge base and a local LLM via Ollama for all operations.

## Results Summary

| Metric | Pipeline A (PageIndex) | Pipeline B (PageIndex+KG) |
|--------|:----------------------:|:-------------------------:|
| Answer Score (1-5) | 3.30 | 3.40 |
| Retrieval Recall | 0.050 | 0.050 |
| Avg Latency (s) | 19.7 | 19.5 |

**Key Finding:** KG expansion helps most on **hard questions** (+1.75 score improvement), suggesting cross-reference links aid multi-step legal reasoning.

## Project Structure

```
ML-Project/
├── README.md                              ← This file
├── task2_data_preprocessing.ipynb         ← Task 2: Data preprocessing
├── task3_model_development.ipynb          ← Task 3: Model development & evaluation
├── CS652_Project_Sheet.pdf
├── Project_Proposal_6809036087.pdf
└── data/
    ├── criminal-datasets.csv              ← Raw data (downloaded)
    ├── eda_overview.png                   ← EDA visualization
    ├── eda_penalties_refs.png             ← Penalty/cross-ref charts
    ├── eda_legal_terms.png                ← Legal terms chart
    ├── preprocessed/                      ← From Task 2
    │   ├── preprocessed_criminal_code.csv ← 444 rows, 21 columns
    │   ├── hierarchy_tree.json            ← Nested tree (root→books→titles→divisions→sections)
    │   ├── entities.json                  ← Per-section entity data
    │   ├── cross_reference_edges.json     ← 360 directed edges
    │   ├── cross_references.gexf          ← Graph for visualization
    │   ├── full_parsed_with_headers.csv
    │   ├── node_summaries.json            ← LLM-generated summaries (459 nodes)
    │   ├── qa_test_dataset.json           ← 20 test questions
    │   └── qa_test_dataset.csv
    └── results/                           ← From Task 3
        ├── pipeline_a_results.json        ← Pipeline A answers + metrics
        ├── pipeline_b_results.json        ← Pipeline B answers + metrics
        ├── evaluation_comparison.csv      ← Side-by-side comparison
        ├── evaluation_summary.json        ← Aggregate scores
        ├── pipeline_comparison.png        ← Visualization charts
        └── per_question_comparison.png    ← Per-question bar chart
```

## Data Source

| Item | Detail |
|------|--------|
| **Source** | [PyThaiNLP/thai-law](https://github.com/PyThaiNLP/thai-law) |
| **Original** | Office of the Council of State (สำนักงานคณะกรรมการกฤษฎีกา) |
| **License** | Public domain (Thai Copyright Act, Article 7) |
| **Sections** | 444 legal sections across 3 books, 15 titles, 33 divisions |

## Thai Criminal Code Structure

```
ประมวลกฎหมายอาญา
├── ภาค 1: บทบัญญัติทั่วไป (General Provisions) — มาตรา 1-106
│   ├── ลักษณะ 1: บทบัญญัติที่ใช้แก่ความผิดทั่วไป (11 หมวด)
│   └── ลักษณะ 2: บทบัญญัติที่ใช้แก่ความผิดลหุโทษ
├── ภาค 2: ความผิด (Offenses) — มาตรา 107-366
│   ├── ลักษณะ 1-12 (specific offense categories)
│   └── Including: ความมั่นคง, การปกครอง, ทรัพย์, ชีวิตและร่างกาย, etc.
└── ภาค 3: ลหุโทษ (Petty Offenses) — มาตรา 367-398
```

## How to Run

### Requirements
- Python 3.11+
- Ollama (for local LLM)
- Packages: `pandas`, `numpy`, `pythainlp`, `matplotlib`, `networkx`, `ollama`, `jupyter`

### Steps

```bash
# 1. Install Ollama and pull a model
ollama pull llama3.2        # 2 GB, fast
ollama pull qwen3:4b        # 2.5 GB, better Thai (optional)

# 2. Install Python packages
pip install pandas numpy pythainlp matplotlib networkx ollama jupyter ipykernel

# 3. Run Task 2 (data preprocessing)
jupyter notebook task2_data_preprocessing.ipynb
# Select kernel and Run All

# 4. Run Task 3 (model development)
jupyter notebook task3_model_development.ipynb
# Select kernel and Run All
# Note: Summary generation takes ~15 minutes on first run (cached after)
```

## Task 2: Data Preprocessing

The notebook `task2_data_preprocessing.ipynb` performs:
1. **Data Acquisition** — Downloads criminal-datasets.csv from PyThaiNLP
2. **Data Cleaning** — Normalizes Thai text, fills missing values
3. **Hierarchy Mapping** — Maps each มาตรา to ภาค/ลักษณะ/หมวด
4. **Text Preprocessing** — Tokenizes with PyThaiNLP
5. **Entity Extraction** — Penalties, cross-references, legal terms
6. **Q&A Test Dataset** — 20 questions (easy/medium/hard)

## Task 3: Model Development

The notebook `task3_model_development.ipynb` implements:

### Architecture
- **PageIndex Tree**: 459-node hierarchy (root → 3 books → 13 titles → 11 divisions → 431 sections) with LLM-generated summaries at each node
- **Light Knowledge Graph**: NetworkX DiGraph with 431 nodes and 359 cross-reference edges
- **Tree Navigation**: LLM selects top-2 branches at each level (beam search)
- **KG Expansion**: 1-hop cross-references + same-division siblings

### Models
- **llama3.2:latest** (2.0 GB) — Used for navigation, summaries, answer generation
- **qwen3:4b** (2.5 GB) — Available but 10-30x slower due to thinking mode

### Evaluation
- 20 test questions across 3 difficulty levels
- Metrics: retrieval recall/precision, answer quality (LLM-as-judge 1-5), latency
- Results by difficulty: Easy (A=3.62, B=3.75), Medium (A=3.62, B=2.88), Hard (A=2.00, B=3.75)
