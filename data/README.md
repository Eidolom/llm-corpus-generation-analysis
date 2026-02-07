# Data Directory

This folder contains input data files for the pipeline.

## Files

### `target_words.txt`
List of target lemmas (one per line). Current study focuses on 10 high-frequency polysemous verbs:
- take, make, hold, keep, break, run, drop, turn, see, look

### TEC Corpus Files
**Private data - not included in this repository.**

Concordance exports from the Textbook English Corpus (TEC) should be placed here. See [../docs/DATA_ACCESS.md](../docs/DATA_ACCESS.md) for access instructions.

Expected format: 
`concordance_textbook_english_corpus.csv`
`semantic_analysis_results.json`
`semantic_analysis_summary.csv`
`textbook_sentences.json`

## Usage

Place your input files here before running the pipeline scripts.
