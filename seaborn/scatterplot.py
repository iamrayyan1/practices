import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.random.rand(100)
y = np.random.rand(100)

# Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=x, y=y, color='green', edgecolor='black', alpha=0.7)

# Adding title and labels
plt.title('Scatter Plot', fontsize=14)
plt.xlabel('X Axis', fontsize=12)
plt.ylabel('Y Axis', fontsize=12)

# Display grid
plt.grid(True)

# Show plot
plt.show()
