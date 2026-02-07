# Setup and Execution Guide

This guide explains how to set up and run the Pragmatic English Corpus analysis pipeline.

## Prerequisites

- **Python 3.8+** (tested with 3.14)
- **Google Gemini API key** (required for data generation and semantic analysis)
- **Git LFS** (recommended for handling large datasets)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

Create a `.env` file in the repository root (or set environment variables):

**Unix/Linux/macOS:**
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

**Windows PowerShell:**
```powershell
$env:GOOGLE_API_KEY = "your_api_key_here"
```

Or copy `.env.example` to `.env` and fill in your API key:
```
GOOGLE_API_KEY=your_api_key_here
```

**Note:** Never commit your `.env` file to version control.

### 4. Initialize Git LFS (optional but recommended)

```bash
git lfs install
```

Git LFS is pre-configured to track CSV, JSON, and PNG files in the `outputs/` directory.

## Project Structure

```
├── src/
│   ├── generators/         # Data generation scripts (SCC)
│   ├── analyzers/          # POS tagging, semantic analysis, visualization
│   └── utils/              # Helper scripts (format converters, API tests)
├── data/                   # Input data (target words, corpus files)
├── outputs/                # Generated results (JSON, CSV, PNG)
├── docs/                   # Documentation (data access, methodology)
└── README.md               # Main project documentation
```

## Execution Workflow

**IMPORTANT:** Always run scripts from the repository root directory.

### Step 1: Generate Synthetic Corpus (SCC)

Generate register-controlled sentences using Gemini API:

```bash
python src/generators/scalable_context_generator.py
```

For larger datasets:

```bash
python src/generators/scalable_context_generator_big_data.py
```

**Output:** `outputs/intermediate_sentences.json`

### Step 2: POS Tagging and Verb Filtering

Filter sentences to retain only those where target words function as verbs:

```bash
python src/analyzers/pos_tagger.py
```

**Input:** `outputs/intermediate_sentences.json`  
**Output:** `outputs/pos_tagging_results.json`

### Step 3: Semantic Classification (LLM-as-Judge)

Classify verb usage as LITERAL or IDIOMATIC:

```bash
python src/analyzers/semantic_analyzer.py
```

For more robust processing with chunking:

```bash
python src/analyzers/semantic_analyzer_V2.py
```

**Input:** `outputs/intermediate_sentences.json`  
**Output:** `outputs/thesis_semantic_data_final.csv`

### Step 4: Statistical Analysis and Visualization

Generate charts and Chi-square test results:

```bash
python src/analyzers/visualizer.py
```

**Input:** `outputs/thesis_semantic_data_final.csv`  
**Outputs:**
- `outputs/results_aggregate_chart.png`
- `outputs/results_heatmap.png`

## Utilities

### Test Gemini API Connection

```bash
python src/utils/debug.py
```

### Convert Sketch Engine Concordances (TEC)

```bash
python src/utils/sketch_engine_converter_nltk.py
```

**Input:** `data/textbook_sentences.json`  
**Output:** `data/semantic_analysis_results.json` (protected)

## Troubleshooting

### Missing NLTK Data

If you encounter errors about missing NLTK resources, they should download automatically on first run. If not, manually download:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

### API Rate Limits

The scripts include deliberate delays (e.g., `time.sleep(2)`) to avoid hitting API rate limits. If you encounter `429 Too Many Requests` errors, increase the sleep duration.

### File Not Found Errors

Ensure you're running scripts from the repository root, not from within subdirectories. All file paths are configured relative to the root.

**Correct:**
```bash
python src/analyzers/pos_tagger.py
```

**Incorrect (will fail):**
```bash
cd src/analyzers
python pos_tagger.py
```

## Data Files

### Input Files

- `data/target_words.txt` - List of target lemmas (one per line)
- `data/textbook_sentences.json` - TEC corpus sentences (not included in repo)

### Output Files

Protected/private data in `data/` (not committed):

- `semantic_analysis_results.json` - TEC analysis results
- `semantic_analysis_summary.csv` - Summary statistics
- `concordance_user_profhippo2_textbook_english_corpus__elen_le_foll_20251206131048.csv` - TEC concordance data

Generated outputs in `outputs/` (tracked with Git LFS):

- `intermediate_sentences.json` - Generated sentences (SCC)
- `pos_tagging_results.json` - Verb-filtered sentences with POS tags
- `thesis_semantic_data_final.csv` - Final dataset with semantic classifications
- `results_aggregate_chart.png` - Register vs. usage stacked bar chart
- `results_heatmap.png` - Word × register heatmap

## Next Steps

See [README.md](README.md) for detailed methodology, research rationale, and limitations.

For access to the TEC corpus, see [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md).
