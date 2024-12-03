import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = np.random.randn(1000)

# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(data, bins=30, kde=True, color='skyblue', edgecolor='black')

# Adding title and labels
plt.title('Histogram', fontsize=14)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Display grid
plt.grid(True)

# Show plot
plt.show()
