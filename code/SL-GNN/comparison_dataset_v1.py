import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data
df = pd.read_csv('Comparison/comparison_modified.csv')

# Set up seaborn style
sns.set_style('whitegrid')
palette = sns.color_palette("muted")

# Define unique values for sources, budgets, and degrees
sources = df['num_sources'].unique()
budgets = df['budget'].unique()
average_degrees = df['avg_degree'].unique()
methods = ['gnn', 'random', 'max_degree_static', 'max_degree_dynamic']


# Function to plot a single case
def plot_case(case_num):
    # Set up the matplotlib figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 18), sharey=True)

    # Plotting for the current case across different sources and budgets
    for i, source in enumerate(sources):
        for j, budget in enumerate(budgets):
            ax = axes[i, j]
            subset = df[(df['case'] == case_num) & (df['num_sources'] == source) & (df['budget'] == budget)]

            # Prepare data for plotting
            means = subset.groupby('avg_degree')[methods].mean()

            # Plot bars without error bars
            means.plot(kind='bar', ax=ax, color=palette)

            ax.set_title(f'Sources: {int(source)}, Budget: {int(budget)}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Average Degree', fontsize=14)
            if j == 0:
                ax.set_ylabel('Infection Rate', fontsize=14)
            if i == 0 and j == 2:
                ax.legend(title="Methods", fontsize=12, title_fontsize=14)
            else:
                ax.legend().set_visible(False)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'./Figures/final_case_data_v1_{case_num}.pdf')
    plt.show()


# Generate figures for each case
for case in [1, 2, 3]:
    plot_case(case)
