import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
AI_INPUT_FILENAME = "outputs\\thesis_pragmatic_data_filtered_synthetic.csv"
TEXTBOOK_INPUT_FILENAME = "outputs\\thesis_semantic_data_final_textbook.csv"
OUTPUT_PNG = "outputs\\results_idiomaticity_heatmaps.png"
OUTPUT_MATRIX_CSV = "outputs\\results_idiomaticity_matrix.csv"
TARGET_LEMMAS = None  # Set to a list like ["make", "take", ...] to force specific lemmas.
TOP_N_LEMMAS = 10


def validate_columns(dataframe, required):
    missing = [c for c in required if c not in dataframe.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: "
            f"{missing}. Found columns: {list(dataframe.columns)}"
        )


# 1. Load and normalize source data
df_ai = pd.read_csv(AI_INPUT_FILENAME)
df_textbook = pd.read_csv(TEXTBOOK_INPUT_FILENAME)
df = pd.concat([df_ai, df_textbook], ignore_index=True)
validate_columns(df, ["Lemma", "Register", "Usage_Category"])

df["Lemma"] = df["Lemma"].astype(str).str.strip().str.lower()
df["Register"] = df["Register"].astype(str).str.strip().str.upper()
df["Usage_Category"] = df["Usage_Category"].astype(str).str.strip().str.upper()

register_aliases = {
    "FORMAL": "HIGH",
    "INSTRUCTIONAL": "NEUTRAL",
    "CASUAL": "LOW",
    "NRC": "TEXTBOOK",
}
df["Register"] = df["Register"].replace(register_aliases)

usage_aliases = {
    "IDIOM": "IDIOMATIC",
    "NON-IDIOMATIC": "LITERAL",
    "NON_IDIOMATIC": "LITERAL",
}
df["Usage_Category"] = df["Usage_Category"].replace(usage_aliases)

allowed_usage = {"IDIOMATIC", "LITERAL"}
df = df[df["Usage_Category"].isin(allowed_usage)].copy()

if df.empty:
    raise ValueError("No rows left after filtering for Usage_Category in {IDIOMATIC, LITERAL}.")

# 2. Keep top lemmas (or caller-provided target lemmas)
if TARGET_LEMMAS is None:
    target_lemmas = (
        df["Lemma"].value_counts().head(TOP_N_LEMMAS).index.tolist()
    )
else:
    target_lemmas = [lemma.lower() for lemma in TARGET_LEMMAS]

df = df[df["Lemma"].isin(target_lemmas)].copy()

# Preserve deterministic lemma order by total frequency
lemma_order = (
    df["Lemma"].value_counts().reindex(target_lemmas).fillna(0).sort_values(ascending=False).index.tolist()
)

register_order = ["TEXTBOOK", "HIGH", "NEUTRAL", "LOW"]
present_registers = [r for r in register_order if r in df["Register"].unique()]
extra_registers = sorted([r for r in df["Register"].unique() if r not in present_registers])
final_register_order = present_registers + extra_registers

# 3. Normalized relative frequencies per (Lemma, Register)
normalized = (
    df.groupby(["Lemma", "Register"])["Usage_Category"]
    .value_counts(normalize=True)
    .rename("Proportion")
    .reset_index()
)

idiomatic_heatmap = (
    normalized[normalized["Usage_Category"] == "IDIOMATIC"]
    .pivot(index="Lemma", columns="Register", values="Proportion")
    .reindex(index=lemma_order, columns=final_register_order)
    .fillna(0)
)

literal_heatmap = (
    normalized[normalized["Usage_Category"] == "LITERAL"]
    .pivot(index="Lemma", columns="Register", values="Proportion")
    .reindex(index=lemma_order, columns=final_register_order)
    .fillna(0)
)

# 4. Save a flat matrix for thesis tables/audit trail
matrix_rows = []
for lemma in lemma_order:
    for register in final_register_order:
        idiomatic_value = idiomatic_heatmap.loc[lemma, register]
        literal_value = literal_heatmap.loc[lemma, register]
        matrix_rows.append(
            {
                "Lemma": lemma,
                "Register": register,
                "Idiomatic_Proportion": idiomatic_value,
                "Literal_Proportion": literal_value,
            }
        )

matrix_df = pd.DataFrame(matrix_rows)
matrix_df.to_csv(OUTPUT_MATRIX_CSV, index=False)

# 5. Plot categorical heatmaps
plt.figure(figsize=(14, 10))
grid = plt.GridSpec(2, 1, hspace=0.35)

ax1 = plt.subplot(grid[0, 0])
sns.heatmap(
    idiomatic_heatmap,
    annot=True,
    cmap="Reds",
    vmin=0,
    vmax=1,
    fmt=".0%",
    cbar_kws={"label": "Idiomatic proportion"},
    ax=ax1,
)
ax1.set_title("Idiomatic Usage Intensity by Lemma and Register")
ax1.set_ylabel("Lemma")
ax1.set_xlabel("Register")

ax2 = plt.subplot(grid[1, 0])
sns.heatmap(
    literal_heatmap,
    annot=True,
    cmap="Blues",
    vmin=0,
    vmax=1,
    fmt=".0%",
    cbar_kws={"label": "Literal proportion"},
    ax=ax2,
)
ax2.set_title("Literal Usage Intensity by Lemma and Register")
ax2.set_ylabel("Lemma")
ax2.set_xlabel("Register")

plt.suptitle(
    "Normalized Usage Profiles (Literal vs Idiomatic)\n"
    "Per Lemma and Register, including Textbook baseline",
    fontsize=14,
    y=0.995,
)
plt.subplots_adjust(top=0.9, hspace=0.35)
plt.savefig(OUTPUT_PNG, dpi=300)

# 6. Print aggregate idiomaticity gap for quick reporting
textbook_rows = df[df["Register"] == "TEXTBOOK"]
ai_rows = df[df["Register"].isin(["HIGH", "NEUTRAL", "LOW"])]

textbook_idiomatic_pct = (
    (textbook_rows["Usage_Category"] == "IDIOMATIC").mean() * 100
    if not textbook_rows.empty
    else float("nan")
)
ai_idiomatic_pct = (
    (ai_rows["Usage_Category"] == "IDIOMATIC").mean() * 100
    if not ai_rows.empty
    else float("nan")
)
gap = ai_idiomatic_pct - textbook_idiomatic_pct

print("\n--- IDIOMATICITY HEATMAP VISUALIZER ---")
print(f"AI input: {AI_INPUT_FILENAME}")
print(f"Textbook input: {TEXTBOOK_INPUT_FILENAME}")
print(f"Lemmas visualized ({len(lemma_order)}): {', '.join(lemma_order)}")
print(f"Registers visualized: {', '.join(final_register_order)}")
print(f"Saved heatmaps: {OUTPUT_PNG}")
print(f"Saved matrix: {OUTPUT_MATRIX_CSV}")
print(f"Textbook idiomatic %: {textbook_idiomatic_pct:.2f}")
print(f"AI idiomatic %: {ai_idiomatic_pct:.2f}")
print(f"AI - Textbook idiomaticity gap: {gap:.2f} percentage points")
