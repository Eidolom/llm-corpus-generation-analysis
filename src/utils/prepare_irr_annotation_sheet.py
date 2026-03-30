from pathlib import Path

import pandas as pd


RANDOM_SEED = 42
SAMPLE_SIZE_PER_GROUP = 100
REQUIRED_COLUMNS = {"Lemma", "Register", "Usage_Category", "Full_Sentence"}


def resolve_existing_path(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find {label} CSV. Looked in:\n{searched}"
    )


def validate_columns(df: pd.DataFrame, label: str, path: Path) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{label} CSV at {path} is missing required columns: {sorted(missing)}"
        )


def load_dataframes(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    outputs_dir = project_root / "outputs"

    # Prefer explicit textbook file if present.
    explicit_textbook_candidates = [
        outputs_dir / "thesis_semantic_data_final_textbook.csv",
        project_root / "thesis_semantic_data_final_textbook.csv",
    ]
    explicit_textbook_path = next(
        (p for p in explicit_textbook_candidates if p.exists()),
        None,
    )

    # Collect semantic CSV candidates and classify by Register values.
    semantic_candidates = sorted(outputs_dir.glob("thesis_semantic_data*.csv"))
    semantic_candidates += sorted(project_root.glob("thesis_semantic_data*.csv"))
    semantic_candidates = list(dict.fromkeys(semantic_candidates))

    textbook_by_register = None
    ai_by_register = None

    for csv_path in semantic_candidates:
        df = pd.read_csv(csv_path)
        if "Register" not in df.columns:
            continue

        regs = set(df["Register"].dropna().astype(str).str.upper().unique())
        if regs == {"TEXTBOOK"} and textbook_by_register is None:
            textbook_by_register = (df, csv_path)
        if {"HIGH", "NEUTRAL", "LOW"}.issubset(regs) and ai_by_register is None:
            ai_by_register = (df, csv_path)

    if explicit_textbook_path is not None:
        textbook_df = pd.read_csv(explicit_textbook_path)
        textbook_path = explicit_textbook_path
    elif textbook_by_register is not None:
        textbook_df, textbook_path = textbook_by_register
    else:
        searched = "\n".join(str(p) for p in explicit_textbook_candidates + semantic_candidates)
        raise FileNotFoundError(
            "Could not locate textbook data. Looked for explicit textbook file and TEXTBOOK-only semantic CSVs in:\n"
            f"{searched}"
        )

    if ai_by_register is not None:
        ai_df, ai_path = ai_by_register
    else:
        ai_path = resolve_existing_path(
            [
                outputs_dir / "thesis_semantic_data_final_2.csv",
                project_root / "thesis_semantic_data_final_2.csv",
            ],
            "AI",
        )
        ai_df = pd.read_csv(ai_path)

    validate_columns(textbook_df, "Textbook", textbook_path)
    validate_columns(ai_df, "AI", ai_path)
    return textbook_df, ai_df, textbook_path, ai_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    textbook_df, ai_df, textbook_path, ai_path = load_dataframes(project_root)

    if len(textbook_df) < SAMPLE_SIZE_PER_GROUP:
        raise ValueError(
            f"Textbook CSV has {len(textbook_df)} rows; need at least {SAMPLE_SIZE_PER_GROUP}."
        )
    if len(ai_df) < SAMPLE_SIZE_PER_GROUP:
        raise ValueError(
            f"AI CSV has {len(ai_df)} rows; need at least {SAMPLE_SIZE_PER_GROUP}."
        )

    textbook_sample = textbook_df.sample(n=SAMPLE_SIZE_PER_GROUP, random_state=RANDOM_SEED)
    ai_sample = ai_df.sample(n=SAMPLE_SIZE_PER_GROUP, random_state=RANDOM_SEED)

    combined_df = pd.concat([textbook_sample, ai_sample], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    combined_df.insert(0, "ID", range(1, len(combined_df) + 1))

    annotation_sheet = combined_df[["ID", "Lemma", "Register", "Full_Sentence"]].copy()
    annotation_sheet["Human_Label"] = ""

    master_key = combined_df[["ID", "Usage_Category"]].copy()

    annotation_out = project_root / "irr_annotation_sheet.csv"
    master_key_out = project_root / "irr_master_key.csv"

    annotation_sheet.to_csv(annotation_out, index=False)
    master_key.to_csv(master_key_out, index=False)

    print(
        "Success: Generated "
        f"{annotation_out.name} and {master_key_out.name} "
        f"with {len(combined_df)} rows (seed={RANDOM_SEED}).\n"
        f"Textbook source: {textbook_path}\n"
        f"AI source: {ai_path}"
    )


if __name__ == "__main__":
    main()
