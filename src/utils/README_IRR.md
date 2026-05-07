# IRR scripts

This folder contains the IRR generator and analysis scripts referenced in the thesis.

Prerequisites
- Python 3.9+
- Install dependencies: `pip install pandas scikit-learn`

Quick usage

1. Generate the IRR annotation sheet and gold key:

```bash
python src/utils/irr_sheet_generator.py
```

Inputs (expected relative to the repository root):
- `Files/thesis_semantic_data_final.csv`
- `Files/thesis_semantic_data_final_2.csv`

Outputs:
- `Files/irr_annotation_sheet.csv` (human annotation sheet)
- `Files/irr_gold_key.csv` (hidden gold key)

2. Compute split IRR results (Cohen's kappa):

```bash
python src/utils/compute_split_irr.py
```

This produces `outputs/irr_split_results.csv` with rows for `ALL`, `NRC`, and `SCC`.

Notes
- The scripts expect the `Files/` directory to contain the thesis CSV inputs. Adjust paths in the scripts if you keep data elsewhere.
- Do not commit generated outputs (e.g., files under `outputs/`) if they contain sensitive or copyrighted content.
