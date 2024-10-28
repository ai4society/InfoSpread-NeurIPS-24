import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
palette = sns.color_palette("muted")

# Creating a DataFrame from the given data
df = pd.read_csv('Comparison/final.csv')

# Set up the matplotlib figure
fig, axes = plt.subplots(3, 3, figsize=(18, 18), sharey=True)

# Dataset types and cases
dataset_types = ['tree', 'erdos-renyi', 'watts-strogatz']
cases = [1, 2, 3]

# Plotting for each case across different network types
for i, case in enumerate(cases):
    for j, dataset in enumerate(dataset_types):
        ax = axes[i, j]
        subset = df[(df['type'] == dataset) & (df['case'] == case)]
        subset.set_index('budget')[['gnn', 'random', 'max_degree_static', 'max_degree_dynamic']].plot(
            kind='bar', ax=ax, color=palette)
        ax.set_title(f'Case {case}: {dataset.replace("-", " ").title()}', fontsize=14)
        ax.set_xlabel('Budget', fontsize=14)
        if j == 0:
            ax.set_ylabel('Infection Rate', fontsize=14)
        ax.legend(title="Methods", fontsize=14, title_fontsize=14)

# Adjust layout
plt.tight_layout()
plt.savefig('./Figures/all_final_case.pdf')
plt.show()