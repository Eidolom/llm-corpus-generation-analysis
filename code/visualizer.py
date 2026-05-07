import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from itertools import combinations
from scipy.stats import chi2_contingency, fisher_exact, fisher_exact

# --- CONFIGURATION ---
INPUT_FILENAME = "thesis_semantic_data_final.csv"

# 1. Load Data
df = pd.read_csv(INPUT_FILENAME)

# Clean up Register names just in case (ensure UPPERCASE)
df['Register'] = df['Register'].str.upper()

# Define specific order for charts (Logical progression)
register_order = ['HIGH', 'NEUTRAL', 'LOW']
usage_order = ['LITERAL', 'IDIOMATIC']

# --- PART 1: THE MAIN FINDING (Stacked Bar Chart) ---
print("\n--- 1. GENERATING AGGREGATE CHART ---")

# Calculate counts and percentages
cross_tab = pd.crosstab(df['Register'], df['Usage_Category'])
cross_tab = cross_tab.reindex(index=register_order, columns=usage_order, fill_value=0)
cross_tab = cross_tab.loc[cross_tab.sum(axis=1) > 0]
cross_tab_prop = cross_tab.div(cross_tab.sum(1), axis=0)

# Plotting
plt.figure(figsize=(10, 6))
# Plot the bars (stacked)
ax = cross_tab_prop.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], width=0.7)

# Formatting
plt.title('Shift in Semantic Usage by Register (Aggregate)', fontsize=14, fontweight='bold')
plt.ylabel('Proportion of Usage (0.0 - 1.0)', fontsize=12)
plt.xlabel('Social Context (Register)', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Semantic Category', loc='upper left', bbox_to_anchor=(1, 1))

# Add percentage labels on the bars
for c in ax.containers:
    labels = [f"{value * 100:.0f}%" if value > 0 else "" for value in c.datavalues]
    ax.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('results_aggregate_chart.png', dpi=300)
print(" Saved 'results_aggregate_chart.png'")

# --- PART 2: THE STATISTICAL TEST ---
print("\n--- 2. RUNNING CHI-SQUARE TEST ---")
# We test if Register and Usage are independent
chi2, p, dof, expected = chi2_contingency(cross_tab)

print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4e}") 
if p < 0.05:
    print(">> RESULT: Statistically Significant! (You can reject the Null Hypothesis)")
else:
    print(">> RESULT: Not Significant.")

# Follow-up analysis for sparse expected frequencies
if (expected < 5).any():
    print("\nExpected frequencies below 5 detected. Running pairwise 2x2 Fisher follow-up tests.")
    pairwise_results = []
    for reg_a, reg_b in combinations(cross_tab.index, 2):
        pair_table = cross_tab.loc[[reg_a, reg_b], usage_order]
        if (pair_table.sum(axis=1) == 0).any() or (pair_table.sum(axis=0) == 0).any():
            continue
        odds_ratio, fisher_p = fisher_exact(pair_table.values)
        pairwise_results.append((reg_a, reg_b, odds_ratio, fisher_p))

    if pairwise_results:
        corrected_alpha = 0.05 / len(pairwise_results)
        print(f"Bonferroni-corrected alpha: {corrected_alpha:.4f}")
        for reg_a, reg_b, odds_ratio, fisher_p in pairwise_results:
            significance = "significant" if fisher_p < corrected_alpha else "not significant"
            print(
                f"Fisher {reg_a} vs {reg_b}: p={fisher_p:.4e}, "
                f"odds_ratio={odds_ratio:.4f} ({significance})"
            )
    else:
        print("Pairwise Fisher follow-up skipped (insufficient variation in 2x2 tables).")

# --- PART 3: THE WORD BREAKDOWN (Heatmap) ---
print("\n--- 3. GENERATING WORD HEATMAP ---")

# Calculate % Idiomatic for every word in every register
pivot = df.groupby(['Lemma', 'Register'])['Usage_Category'].value_counts(normalize=True).unstack(fill_value=0)

# We only care about the "IDIOMATIC" column for the heatmap
if 'IDIOMATIC' in pivot.columns:
    heatmap_data = pivot['IDIOMATIC'].unstack().reindex(columns=register_order)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="Blues", fmt=".0%", cbar_kws={'label': '% Idiomatic Usage'})
    plt.title('Heatmap: Intensity of Idiomatic Usage by Word', fontsize=14)
    plt.ylabel('Target Lemma')
    plt.xlabel('Register')
    plt.tight_layout()
    plt.savefig('results_heatmap.png', dpi=300)
    print("Saved 'results_heatmap.png'")
else:
    print("Could not generate heatmap (Label 'IDIOMATIC' missing?)")

print("\nDONE. Open the PNG files to see your thesis results.")