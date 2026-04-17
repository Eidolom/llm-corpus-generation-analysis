"""
Split IRR analysis for NRC vs SCC.

This script compares human labels to model labels and reports:
- aggregate Cohen's kappa
- NRC-only Cohen's kappa
- SCC-only Cohen's kappa

Expected inputs
---------------
- Files/irr_annotation_sheet - irr_annotation_sheet.csv.csv
  (human labels)
- Files/irr_master_key.csv (optional)
  (model labels + metadata)

If a master-key file is unavailable, the script can read a merged sheet
that already contains both human and model labels.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import re

import pandas as pd
from sklearn.metrics import cohen_kappa_score


VALID_LABELS = {"LITERAL", "IDIOMATIC"}


@dataclass
class SplitResult:
    group: str
    n_valid: int
    raw_agreement: float | None
    kappa: float | None


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_label(value: object) -> str:
    label = "" if value is None else str(value)
    return label.strip().upper()


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def compute_group(df: pd.DataFrame, group_name: str) -> SplitResult:
    clean = df.copy()
    clean["human_label"] = clean["human_label"].map(normalize_label)
    clean["model_label"] = clean["model_label"].map(normalize_label)

    keep = (
        clean["human_label"].isin(VALID_LABELS)
        & clean["model_label"].isin(VALID_LABELS)
    )
    clean = clean[keep]

    if clean.empty:
        return SplitResult(group_name, 0, None, None)

    raw = (clean["human_label"] == clean["model_label"]).mean()
    kappa = cohen_kappa_score(clean["human_label"], clean["model_label"])
    return SplitResult(group_name, int(len(clean)), float(raw), float(kappa))


def kappa_interpretation(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.00:
        return "poor"
    if value < 0.20:
        return "slight"
    if value < 0.40:
        return "fair"
    if value < 0.60:
        return "moderate"
    if value < 0.80:
        return "substantial"
    return "almost perfect"


def load_input_frames(project_root: Path) -> pd.DataFrame:
    files_dir = project_root / "Files"

    annotation_path = (
        files_dir / "irr_annotation_sheet - irr_annotation_sheet.csv.csv"
    )
    master_key_path = files_dir / "irr_master_key.csv"

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Missing annotation file: {annotation_path}"
        )

    ann = pd.read_csv(annotation_path)

    human_col = first_existing(
        ann.columns.tolist(),
        ["Human_Label", "human_label", "human", "label_human"],
    )
    sent_col = first_existing(
        ann.columns.tolist(),
        ["Full_Sentence", "full_sentence", "sentence", "text"],
    )

    if human_col is None:
        raise ValueError("Could not find human label column.")

    ann = ann.rename(columns={human_col: "human_label"})

    if master_key_path.exists() and sent_col is not None:
        key = pd.read_csv(master_key_path)
        key_sent = first_existing(
            key.columns.tolist(),
            ["Full_Sentence", "full_sentence", "sentence", "text"],
        )
        key_model = first_existing(
            key.columns.tolist(),
            ["model_label", "Model_Label", "ai_label", "label_model"],
        )
        key_group = first_existing(
            key.columns.tolist(),
            ["source_group", "Source_Group", "group", "corpus"],
        )

        if key_sent and key_model and key_group and sent_col:
            ann["_sent_key"] = ann[sent_col].map(normalize_text)
            key["_sent_key"] = key[key_sent].map(normalize_text)
            keep_cols = ["_sent_key", key_model, key_group]
            key = key[keep_cols].drop_duplicates("_sent_key")
            key = key.rename(
                columns={
                    key_model: "model_label",
                    key_group: "source_group",
                }
            )
            merged = ann.merge(key, on="_sent_key", how="left")
            return merged

    model_col = first_existing(
        ann.columns.tolist(),
        ["Model_Label", "model_label", "ai_label", "label_model"],
    )
    group_col = first_existing(
        ann.columns.tolist(),
        ["Source_Group", "source_group", "group", "corpus"],
    )

    if model_col is None or group_col is None:
        raise ValueError(
            "Need either Files/irr_master_key.csv or columns "
            "for model labels and source group in annotation sheet."
        )

    ann = ann.rename(columns={model_col: "model_label", group_col: "source_group"})
    return ann


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    df = load_input_frames(project_root)

    if "source_group" not in df.columns:
        raise ValueError("Missing source_group column.")
    if "model_label" not in df.columns:
        raise ValueError("Missing model_label column.")

    df["source_group"] = df["source_group"].astype(str).str.upper().str.strip()

    aggregate = compute_group(df, "ALL")
    nrc = compute_group(df[df["source_group"] == "NRC"], "NRC")
    scc = compute_group(df[df["source_group"] == "SCC"], "SCC")

    rows = [aggregate, nrc, scc]

    print("=" * 64)
    print(f"{'Group':<10}{'n':>8}{'Raw Agr.':>12}{'Kappa':>10}  Interpretation")
    print("-" * 64)
    for r in rows:
        if r.kappa is None:
            print(f"{r.group:<10}{'-':>8}{'-':>12}{'-':>10}  n/a")
            continue
        raw_pct = f"{r.raw_agreement * 100:.1f}%"
        print(
            f"{r.group:<10}{r.n_valid:>8}{raw_pct:>12}"
            f"{r.kappa:>10.3f}  {kappa_interpretation(r.kappa)}"
        )
    print("=" * 64)

    out_dir = project_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "irr_split_results.csv"

    out_df = pd.DataFrame(
        {
            "group": [r.group for r in rows],
            "n_valid": [r.n_valid for r in rows],
            "raw_agreement": [r.raw_agreement for r in rows],
            "kappa": [r.kappa for r in rows],
            "interpretation": [kappa_interpretation(r.kappa) for r in rows],
        }
    )
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()