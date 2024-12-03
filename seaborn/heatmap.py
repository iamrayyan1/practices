import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = np.random.rand(10, 12)

# Heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(data, annot=True, cmap='coolwarm', cbar=True, linewidths=0.5)

# Adding title
plt.title('Heatmap', fontsize=14)

# Show plot
plt.show()
