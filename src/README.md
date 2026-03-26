# Source Code

All Python scripts for the analysis pipeline.

## Structure

### `generators/`
Scripts for generating the Synthetic Control Corpus (SCC):
- `scalable_context_generator.py` - Small-scale register-controlled sentence generation
- `scalable_context_generator_big_data.py` - Batch generation for large corpora
- `simple_data_collector.py` - Minimal test/pilot generation

### `analyzers/`
Scripts for POS filtering, semantic classification, and statistical analysis:
- `pos_tagger.py` - **Main POS tagger:** NLTK-based lemmatization + verb filter + context window extraction
- `semantic_analyzer.py` - LLM-as-judge classification (single-batch)
- `semantic_analyzer_V2.py` - Chunked LLM classification (robust for large inputs)
- `scalable_pos_tagger.py` - LLM-based POS tagging (experimental alternative)
- `visualizer.py` - Statistical tests and visualization generation

### `utils/`
Utility scripts:
- `sketch_engine_converter_nltk.py` - Convert TEC concordance exports to JSON
- `debug.py` - API connectivity test

## Running Scripts

All scripts should be run from the repository root directory:

```powershell
# Example: Generate sentences
python src/generators/scalable_context_generator.py

# Example: Filter verbs
python src/analyzers/pos_tagger.py

# Example: Classify semantics
python src/analyzers/semantic_analyzer.py
```

**API Key Setup:**
The scripts automatically load your API key from the `.env` file in the project root. To configure it:

1. Copy `.env.example` to `.env`
2. Add your Gemini API key: `GOOGLE_API_KEY=your_key_here`
3. Run scripts normally—the venv handles authentication
