"""
Generate blinded IRR annotation material for the thesis workflow.

Outputs:
1) Files/irr_annotation_sheet.csv
   - Sheet used for manual coding (no model labels or register metadata).
2) Files/irr_gold_key.csv
   - Hidden key containing model labels for later agreement calculation.

Design:
- 100 NRC items (TEXTBOOK)
- 100 SCC items (stratified across HIGH / NEUTRAL / LOW)
- Deterministic random seed for reproducibility
"""

import pandas as pd


# --- CONFIGURATION ---
NRC_INPUT_FILE = "Files/thesis_semantic_data_final.csv"
SCC_INPUT_FILE = "Files/thesis_semantic_data_final_2.csv"

ANNOTATION_OUTPUT_FILE = "Files/irr_annotation_sheet.csv"
KEY_OUTPUT_FILE = "Files/irr_gold_key.csv"

NRC_SAMPLE_SIZE = 100
SCC_SAMPLE_SIZE = 100
RANDOM_SEED = 42

REQUIRED_COLUMNS = ["Lemma", "Register", "Mood", "Usage_Category", "Full_Sentence"]
SCC_REGISTER_ORDER = ["HIGH", "NEUTRAL", "LOW"]


def validate_columns(df: pd.DataFrame, source_name: str) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def sample_nrc(nrc_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    nrc_pool = nrc_df[nrc_df["Register"].astype(str).str.upper() == "TEXTBOOK"].copy()
    if len(nrc_pool) < n:
        raise ValueError(f"NRC pool too small: requested {n}, available {len(nrc_pool)}")

    sample = nrc_pool.sample(n=n, random_state=seed).copy()
    sample["Source_Group"] = "NRC"
    return sample


def sample_scc_stratified(scc_df: pd.DataFrame, n_total: int, seed: int) -> pd.DataFrame:
    scc_df = scc_df.copy()
    scc_df["Register"] = scc_df["Register"].astype(str).str.upper()

    missing_registers = [reg for reg in SCC_REGISTER_ORDER if reg not in set(scc_df["Register"])]
    if missing_registers:
        raise ValueError(f"SCC data is missing required registers: {missing_registers}")

    base = n_total // len(SCC_REGISTER_ORDER)
    remainder = n_total % len(SCC_REGISTER_ORDER)

    allocations = {
        register: base + (1 if idx < remainder else 0)
        for idx, register in enumerate(SCC_REGISTER_ORDER)
    }

    sampled_chunks = []
    for idx, register in enumerate(SCC_REGISTER_ORDER):
        n_register = allocations[register]
        register_pool = scc_df[scc_df["Register"] == register]

        if len(register_pool) < n_register:
            raise ValueError(
                f"SCC pool for {register} too small: requested {n_register}, available {len(register_pool)}"
            )

        sampled_chunk = register_pool.sample(n=n_register, random_state=seed + idx).copy()
        sampled_chunks.append(sampled_chunk)

    sample = pd.concat(sampled_chunks, ignore_index=True)
    sample["Source_Group"] = "SCC"
    return sample


def build_outputs(nrc_sample: pd.DataFrame, scc_sample: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([nrc_sample, scc_sample], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    combined["IRR_ID"] = [f"IRR_{idx:04d}" for idx in range(1, len(combined) + 1)]
    combined["Model_Label"] = combined["Usage_Category"].astype(str).str.upper()

    annotation_sheet = combined[["IRR_ID", "Lemma", "Full_Sentence"]].rename(
        columns={
            "IRR_ID": "irr_id",
            "Lemma": "lemma",
            "Full_Sentence": "sentence",
        }
    )
    annotation_sheet["human_label"] = ""
    annotation_sheet["notes"] = ""

    gold_key = combined[
        [
            "IRR_ID",
            "Source_Group",
            "Register",
            "Mood",
            "Lemma",
            "Model_Label",
            "Full_Sentence",
        ]
    ].rename(
        columns={
            "IRR_ID": "irr_id",
            "Source_Group": "source_group",
            "Register": "register",
            "Mood": "mood",
            "Lemma": "lemma",
            "Model_Label": "model_label",
            "Full_Sentence": "sentence",
        }
    )

    return annotation_sheet, gold_key


def main() -> None:
    print("--- Generating IRR annotation sheet ---")

    nrc_df = pd.read_csv(NRC_INPUT_FILE)
    scc_df = pd.read_csv(SCC_INPUT_FILE)

    validate_columns(nrc_df, "NRC input")
    validate_columns(scc_df, "SCC input")

    nrc_sample = sample_nrc(nrc_df, n=NRC_SAMPLE_SIZE, seed=RANDOM_SEED)
    scc_sample = sample_scc_stratified(scc_df, n_total=SCC_SAMPLE_SIZE, seed=RANDOM_SEED)

    annotation_sheet, gold_key = build_outputs(nrc_sample, scc_sample, seed=RANDOM_SEED)

    annotation_sheet.to_csv(ANNOTATION_OUTPUT_FILE, index=False)
    gold_key.to_csv(KEY_OUTPUT_FILE, index=False)

    print(f"Saved annotation sheet: {ANNOTATION_OUTPUT_FILE}")
    print(f"Saved gold key: {KEY_OUTPUT_FILE}")
    print(f"Total IRR sample size: {len(annotation_sheet)}")

    source_counts = gold_key["source_group"].value_counts().to_dict()
    register_counts = gold_key["register"].value_counts().to_dict()
    print(f"Source balance: {source_counts}")
    print(f"Register balance: {register_counts}")


if __name__ == "__main__":
    main()
