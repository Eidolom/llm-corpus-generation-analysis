# Register and Idiomaticity in Verb Usage: A Corpus-Driven Comparison of Textbook English and LLM-Generated Sentences

Thesis-oriented NLP pipeline for comparing authentic textbook English (Natural Reference Corpus) with LLM-generated sentences (Synthetic Control Corpus). This repository implements a staged pipeline for sentence generation, POS-based verb filtering, and LLM-as-judge semantic classification to analyze register variation and idiomatic usage patterns.

## Overview

This study uses a quantitative corpus-based approach to compare two datasets: a **Natural Reference Corpus (NRC)** of authentic textbook English from the Textbook English Corpus (TEC) and a **Synthetic Control Corpus (SCC)** of LLM-generated sentences. The goal is to examine whether AI-generated examples differ systematically from authentic pedagogical materials in register and idiomaticity.

The workflow implements an ETL (Extract, Transform, Load) pipeline with strict POS-based filtering and LLM-as-judge semantic classification to ensure reproducibility and analytical rigor.

## Methodology

### 3.1 Corpus Compilation

#### Natural Reference Corpus (NRC)
The NRC is compiled from the **Textbook English Corpus (TEC)**, a specialized corpus of EFL textbook materials (Le Foll 2021a, 2021b). The TEC contains materials from 33 ESL/EFL textbooks published between 2006–2018 by major publishers (Oxford, Cambridge, Klett, Cornelsen, Nathan, Bordas, Richmond), representing both European markets and international coursebooks.

**Sampling procedure:**
- **CEFR filtering:** Subcorpus limited to B1/B2 level materials
- **Target lemma extraction:** Concordance searches for 10 high-frequency polysemous verbs (take, make, hold, keep, break, run, drop, turn, see, look)
- **Verb-only filtering:** NLTK POS-tagging to retain only instances where the target lemma appears as a verb

#### Synthetic Control Corpus (SCC)
Generated using `scalable_context_generator.py`, which prompts Gemini Flash 2.5 to produce sentences for the same 10 target lemmas under three register conditions:
1. **Register HIGH:** Formal/Workplace (Questions)
2. **Register NEUTRAL:** Direct/Instructions (Imperatives)
3. **Register LOW:** Casual/Friends (Statements)

Each sentence includes metadata (lemma, register, mood) for register-based comparison.

### 3.2 Computational Workflow

#### 3.2.1 POS-Tagging and Verb Filter
**Problem:** Many target items are functionally ambiguous (e.g., "run" as noun vs. verb). String-based searches would include irrelevant tokens like "a morning run."

**Solution:** `pos_tagger.py` implements a strict gatekeeper:
1. Tokenizes sentences using NLTK
2. Applies the NLTK Averaged Perceptron Tagger
3. Lemmatizes each token using WordNet
4. **Lemma-based matching:** Checks if the lemmatized form matches the target lemma (case-insensitive) AND is tagged as a verb (VB, VBD, VBG, VBN, VBP, VBZ)
5. Retains only sentences with verified verb usage

This ensures inflected forms (runs, ran, running) are correctly identified and non-verb uses are excluded.

#### 3.2.2 Quantifying Idiomaticity: LLM as a Judge
`semantic_analyzer.py` (and the chunked variant `semantic_analyzer_V2.py`) implements **LLM-as-judge** classification (Zheng et al. 2023) to tag each sentence as:
- **LITERAL:** Physical/core meaning
- **IDIOMATIC:** De-lexicalized, metaphorical, or fixed phrase usage

**Rationale:** Idiomaticity extends beyond particle verbs (e.g., "run a company" is metaphorical but has no particle). Rule-based approaches would miss such cases. The LLM makes constrained semantic decisions by processing filtered sentences in small chunks with strict JSON output formatting.

**Key variables:**
- **Dependent variable:** Usage_Category (LITERAL vs. IDIOMATIC)
- **Grouping variable:** Register (HIGH / NEUTRAL / LOW)

### 3.3 Statistical Analysis
The primary statistical test is a **Chi-Square test of independence (χ²)** to assess whether Register and Usage_Category are independent (α = 0.05). For cells with low expected frequencies, **Fisher's Exact Test** is additionally applied.

`visualizer.py` produces:
- Aggregate stacked bar chart by register
- Per-lemma heatmap of idiomatic rates

## Repository Layout

```
project/
├── src/
│   ├── generators/       # SCC data generation
│   ├── analyzers/        # POS filtering, semantic classification, stats
│   └── utils/            # Conversion and testing utilities
├── data/                 # Input files (target_words.txt, TEC exports)
├── outputs/              # Generated results (JSON, CSV, plots)
├── docs/                 # Additional documentation
├── .gitignore
├── .gitattributes        # Git LFS configuration
├── .env.example          # Environment variable template
├── README.md
└── requirements.txt
```

### Core Pipeline Scripts
- **Data Generation (SCC)** - `src/generators/`
  - `scalable_context_generator.py`: Generate register-controlled sentences (small-scale)
  - `scalable_context_generator_big_data.py`: Batch generation for large corpora
  - `simple_data_collector.py`: Minimal test/pilot generation
- **TEC Data Processing (NRC)** - `src/utils/`
  - `sketch_engine_converter_nltk.py`: Convert TEC concordance exports to JSON
- **POS Filtering (Both Corpora)** - `src/analyzers/`
  - `pos_tagger.py`: Lemmatize + verb-filter + context window extraction (NLTK-based, production)
  - `scalable_pos_tagger.py`: LLM-based POS tagging (alternative/experimental)
- **Semantic Classification (AI Judge)** - `src/analyzers/`
  - `semantic_analyzer.py`: Single-batch LLM classification
  - `semantic_analyzer_V2.py`: Chunked processing for robustness
- **Statistical Analysis** - `src/analyzers/`
  - `visualizer.py`: Chi-square tests, plots, and summary statistics
- **Utilities** - `src/utils/`
  - `debug.py`: API connectivity test

## Data Flow

### Pipeline A: Synthetic Control Corpus (SCC)
1. **Generate sentences**
   - Input: `data/target_words.txt` (10 target lemmas)
   - Script: `src/generators/scalable_context_generator.py` or `src/generators/scalable_context_generator_big_data.py`
   - Output: `outputs/intermediate_sentences.json` (register-controlled sentences)
2. **Verb filtering**
   - Input: `outputs/intermediate_sentences.json`
   - Script: `src/analyzers/pos_tagger.py`
   - Output: `outputs/pos_tagging_results.json` (verb-only sentences)
3. **Semantic classification**
   - Input: `outputs/intermediate_sentences.json` (or filtered results)
   - Script: `src/analyzers/semantic_analyzer.py` or `src/analyzers/semantic_analyzer_V2.py`
   - Output: `outputs/thesis_semantic_data_final.csv` (LITERAL/IDIOMATIC labels)
4. **Statistical analysis**
   - Input: `outputs/thesis_semantic_data_final.csv`
   - Script: `src/analyzers/visualizer.py`
   - Outputs: Chi-square results, plots (`outputs/results_aggregate_chart.png`, `outputs/results_heatmap.png`)

### Pipeline B: Natural Reference Corpus (NRC)
1. **Export TEC concordances**
   - Export from Sketch Engine (see [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md))
2. **Convert to JSON**
   - Script: `src/utils/sketch_engine_converter_nltk.py`
   - Output: `data/textbook_sentences.json`
3. **Apply verb filter + semantic classification** (same as Pipeline A steps 2–4)

## Setup

### 1) Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) API key (no secrets in repo)

Set your API key in the environment. Do not commit real keys.

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

Windows PowerShell:

```powershell
$env:GOOGLE_API_KEY = "your_api_key_here"
```

You can optionally copy `.env.example` to `.env` for local use (do not commit it).

## Git LFS (Large Files)

This repo includes large datasets and outputs. Install Git LFS and ensure it tracks the patterns in `.gitattributes`:

```bash
git lfs install
git lfs track "*.csv"
git lfs track "*.json"
git lfs track "*.png"
```

## Usage

### Target Lemmas
The current study focuses on 10 high-frequency polysemous verbs: **take, make, hold, keep, break, run, drop, turn, see, look**. These are specified in `data/target_words.txt` (one lemma per line).

### Generate data (SCC)

```bash
python src/generators/scalable_context_generator.py
```

For larger corpora:

```bash
python src/generators/scalable_context_generator_big_data.py
```

### Gatekeep (POS + lemma match)

```bash
python src/analyzers/pos_tagger.py
```

### AI judge (semantic classification)

```bash
python src/analyzers/semantic_analyzer.py
```

Chunked variant (more robust for large inputs):

```bash
python src/analyzers/semantic_analyzer_V2.py
```

### Visualizations

```bash
python src/analyzers/visualizer.py
```

## Outputs

Common outputs in `outputs/` (tracked with Git LFS):

- `intermediate_sentences.json`
- `pos_tagging_results.json`
- `thesis_semantic_data_final.csv`
- `results_aggregate_chart.png`
- `results_heatmap.png`

Private/corpus data in `data/` (not committed, access on request):

- `semantic_analysis_results.json`
- `semantic_analysis_summary.csv`
- `concordance_user_profhippo2_textbook_english_corpus__elen_le_foll_20251206131048.csv`

See [outputs/README.md](outputs/README.md) for details.

## Data Access (TEC Corpus)

The TEC corpus files are private and not published in this repository. Access is provided on request only. See [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) for the request process.

## Reproducibility Notes

### Version Control
- **LLM model:** Gemini Flash 2.5 (via `google-generativeai` Python SDK)
- **POS tagger:** NLTK Averaged Perceptron Tagger
- **Python:** 3.8+ recommended

### Known Variability
- LLM outputs may vary across runs due to non-deterministic generation. For exact replication, use the same model version and API endpoint.
- Rate limits and latency are handled with explicit sleeps in the scripts.
- NLTK data (tokenizer, tagger, WordNet) should be downloaded automatically on first run.

### Reproducing Results
1. Use the same `target_words.txt` file
2. Set consistent CEFR filtering parameters (B1/B2) if using TEC
3. Run `pos_tagger.py` before semantic analysis to ensure verb-only filtering
4. Use the same significance level (α = 0.05) in statistical tests

## Limitations

### 4.1 Model Bias (Technical)
This study uses Gemini 2.5 Flash for both data generation and semantic evaluation for API cost efficiency. While the model excels at high-volume processing, "Flash" models are optimized for latency rather than deep reasoning. The model may show a **"simplicity bias,"** potentially under-reporting complex idiomatic usage compared to larger reasoning models (e.g., Claude Opus 4.5 or GPT-5.2). Future iterations could use a multi-LLM design to cross-validate results across different models.

### 4.2 Proxy Measures (Methodological)

#### Contraction Counting
The operationalization of linguistic variables relies on proxy measures. While "structural reduction" (Section 2.1.3) includes contractions and ellipsis, the current workflow quantifies it via a simple apostrophe-based contraction count (`pos_tagger.py`). This count does not distinguish between:
- Clitics (e.g., "I'm")
- Possessive markers (e.g., "John's")
- Apostrophes in quoted material

Ellipsis is not included in the present implementation.

#### Register-Mood Coupling
The study couples register with grammatical mood:
- **HIGH** = Questions
- **NEUTRAL** = Imperatives
- **LOW** = Statements

This means observed differences in idiomaticity cannot be cleanly attributed to register (formal vs. casual) because they may instead be caused by mood-specific syntax. For example, interrogatives often introduce auxiliaries (could, will) in subject-auxiliary inversion. A more robust design would decouple register and mood by crossing conditions (e.g., HIGH-Q workplace questions, HIGH-S workplace statements, LOW-Q casual questions, LOW-S casual statements). This would increase corpus size but allow separating register effects from mood effects.

### 4.3 Model Self-Preference
A potential concern arises from the **"LLM-as-a-judge" methodology.** As the same model family (Gemini) is used to both generate the Synthetic Control Corpus and classify its semantic usage, there is a risk of **self-preference bias.** The model may be inclined to rate its own generations as contextually appropriate. Future research should employ an independent "judge" model to ensure independence during the evaluation process.

### 4.4 Corpus Representativeness (NRC Scope)
The NRC, derived from the Textbook English Corpus (TEC), represents a specific subset of textbook English: materials explicitly designed for B1/B2 learners. This has two implications:

**CEFR Level Generalization:**
Findings may not generalize to other CEFR levels. A1/A2 materials may show even higher explicitness (lower idiomaticity), while C1/C2 materials may contain more authentic registers.

**Temporal Scope:**
The TEC compilation date range (2006–2018) means the NRC reflects textbook design practices from that period. Le Foll (2021b) found that textbook English shows distinct register characteristics compared to general English corpora, particularly in its pedagogical framing and explicit grammatical structures.

If current ELT materials (2020–2025) show narrower gaps between textbook and authentic language, this would support H2 (Corpus-Informed Pedagogy Hypothesis) and suggest that LLMs are converging with an already-shifting policy in learning material design. Conversely, if the 2006–2018 TEC materials already show substantial idiomaticity, this would challenge the assumption that textbooks avoid phraseological variation.

Despite these challenges, the B1/B2 focus is methodologically appropriate because this proficiency band represents the majority of adult ESL learners and is the primary target for intermediate textbooks where the difference between "teachability" and "authenticity" has arguably the most impact.

### 4.5 Statistical Constraints
- **Low frequency cells:** Chi-square test assumes sufficient cell counts; lemma × register × category combinations with low frequencies may not meet test assumptions (Fisher's Exact Test is used as a fallback).
- **Lemma coverage:** Analysis limited to 10 high-frequency polysemous verbs; findings may not generalize to other verb classes or less common items.

### 4.6 Data Quality
- **POS tagger accuracy:** NLTK Averaged Perceptron Tagger has known error rates; some ambiguous cases may be misclassified.
- **LLM semantic judgments:** LITERAL vs. IDIOMATIC classifications are made by the LLM and may not align with all human annotator judgments. No inter-rater reliability is computed.

### 4.7 Practical Constraints
- **API costs:** Large datasets incur API costs. The verb filter is strict to reduce unnecessary calls, but scaling to hundreds of lemmas requires budget planning.
- **Rate limits:** Explicit sleep intervals are implemented to respect API rate limits; processing large corpora is time-intensive.
- **LLM variability:** Model outputs are probabilistic and may vary across runs. For consistency, re-run with the same model version.

## Security

- API keys are never stored in code.
- Use environment variables or a local `.env` file (ignored by git).
- Rotate any previously exposed keys.

## References

- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.
- Le Foll, E. (2021a). The Textbook English Corpus (TEC). [Corpus dataset].
- Le Foll, E. (2021b). Register variation in EFL textbooks: A corpus-based investigation. *Applied Linguistics*.
- Loper, E., & Bird, S. (2002). NLTK: The Natural Language Toolkit. *Proceedings of the ACL-02 Workshop on Effective Tools and Methodologies for Teaching Natural Language Processing and Computational Linguistics*, 63–70.
- Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *arXiv preprint*.

