import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURATION (kept aligned with visualizer_idiomaticity_heatmap.py) ---
AI_INPUT_FILENAME = "outputs\\thesis_pragmatic_data_filtered_synthetic.csv"
TEXTBOOK_INPUT_FILENAME = "outputs\\thesis_semantic_data_final_textbook.csv"
AI_FALLBACK_FILENAME = "outputs\\thesis_semantic_data_final_2.csv"
OUTPUT_PNG = "outputs\\results_aggregate_register_proportions.png"
OUTPUT_CSV = "outputs\\results_aggregate_register_proportions.csv"


def normalize_columns(dataframe):
    """Standardize common column-name variants into a shared schema."""
    rename_map = {
        "lemma": "Lemma",
        "register": "Register",
        "usage_category": "Usage_Category",
    }
    return dataframe.rename(columns=rename_map)


def normalize_values(dataframe):
    dataframe["Register"] = dataframe["Register"].astype(str).str.strip().str.upper()
    dataframe["Usage_Category"] = dataframe["Usage_Category"].astype(str).str.strip().str.upper()

    register_aliases = {
        "FORMAL": "HIGH",
        "INSTRUCTIONAL": "NEUTRAL",
        "CASUAL": "LOW",
        "NRC": "TEXTBOOK",
    }
    usage_aliases = {
        "IDIOM": "IDIOMATIC",
        "NON-IDIOMATIC": "LITERAL",
        "NON_IDIOMATIC": "LITERAL",
    }

    dataframe["Register"] = dataframe["Register"].replace(register_aliases)
    dataframe["Usage_Category"] = dataframe["Usage_Category"].replace(usage_aliases)
    return dataframe


def validate_required_columns(dataframe, source_name):
    required_columns = ["Register", "Usage_Category"]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns {missing}. "
            f"Found columns: {list(dataframe.columns)}"
        )


# 1) Load and harmonize the same final-dataset sources
ai_df = pd.read_csv(AI_INPUT_FILENAME)
textbook_df = pd.read_csv(TEXTBOOK_INPUT_FILENAME)

ai_df = normalize_columns(ai_df)
textbook_df = normalize_columns(textbook_df)

active_ai_source = AI_INPUT_FILENAME
if "Usage_Category" not in ai_df.columns and Path(AI_FALLBACK_FILENAME).exists():
    fallback_df = normalize_columns(pd.read_csv(AI_FALLBACK_FILENAME))
    if "Usage_Category" in fallback_df.columns:
        ai_df = fallback_df
        active_ai_source = AI_FALLBACK_FILENAME
        print(
            f"[info] AI input '{AI_INPUT_FILENAME}' has no Usage_Category; "
            f"using fallback '{AI_FALLBACK_FILENAME}'."
        )

validate_required_columns(ai_df, "AI input")
validate_required_columns(textbook_df, "Textbook input")

combined_df = pd.concat([ai_df, textbook_df], ignore_index=True)
combined_df = normalize_values(combined_df)

combined_df = combined_df[combined_df["Usage_Category"].isin(["LITERAL", "IDIOMATIC"])].copy()
if combined_df.empty:
    raise ValueError("No valid rows found after filtering Usage_Category to LITERAL/IDIOMATIC.")

# 2) Aggregate proportions by register
counts = pd.crosstab(combined_df["Register"], combined_df["Usage_Category"])
register_order = ["TEXTBOOK", "HIGH", "NEUTRAL", "LOW"]
ordered_index = [reg for reg in register_order if reg in counts.index]
ordered_index.extend([reg for reg in counts.index if reg not in ordered_index])
counts = counts.reindex(ordered_index, fill_value=0)

proportions = counts.div(counts.sum(axis=1), axis=0).fillna(0)
for category in ["LITERAL", "IDIOMATIC"]:
    if category not in proportions.columns:
        proportions[category] = 0.0
proportions = proportions[["LITERAL", "IDIOMATIC"]]

# 3) Save tabular output
summary_df = proportions.reset_index().rename(columns={"index": "Register"})
summary_df.to_csv(OUTPUT_CSV, index=False)

# 4) Plot stacked bar chart
ax = proportions.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    color=["#4C78A8", "#E45756"],
    width=0.7,
)

ax.set_title("Aggregate proportion of LITERAL vs. IDIOMATIC usages by register (Final Dataset)")
ax.set_xlabel("Register")
ax.set_ylabel("Proportion")
ax.set_ylim(0, 1)
ax.legend(title="Usage Category", loc="upper left", bbox_to_anchor=(1, 1))

for container in ax.containers:
    labels = [f"{value * 100:.0f}%" if value > 0 else "" for value in container.datavalues]
    ax.bar_label(container, labels=labels, label_type="center", color="white", fontweight="bold")

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)

print("\n--- AGGREGATE REGISTER PROPORTIONS ---")
print(f"AI input: {active_ai_source}")
print(f"Textbook input: {TEXTBOOK_INPUT_FILENAME}")
print(f"Saved chart: {OUTPUT_PNG}")
print(f"Saved summary: {OUTPUT_CSV}")
print(summary_df.to_string(index=False))
