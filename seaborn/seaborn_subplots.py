
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Generate a sample DataFrame
np.random.seed(10)
data = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C'], 300),
    'Values1': np.random.normal(loc=50, scale=15, size=300),
    'Values2': np.random.normal(loc=30, scale=10, size=300)
})

# Create a FacetGrid in Seaborn
g = sns.FacetGrid(data, col="Category", col_wrap=2, height=4)
g.map(sns.histplot, "Values1", kde=True, color="skyblue")
g.set_titles("{col_name} Category")
g.set_axis_labels("Values", "Frequency")
plt.show()

# Using matplotlib subplots with Seaborn
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Boxplot in the first subplot
sns.boxplot(x="Category", y="Values1", data=data, ax=axs[0], palette="Set2")
axs[0].set_title("Boxplot of Values1 by Category")

# Scatterplot in the second subplot
sns.scatterplot(x="Values1", y="Values2", hue="Category", data=data, ax=axs[1], palette="Set1")
axs[1].set_title("Scatterplot of Values1 vs Values2")

plt.tight_layout()
plt.show()
