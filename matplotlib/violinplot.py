import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Sample data
data = np.random.rand(100, 3)

# Violin plot
plt.figure(figsize=(8, 5))
sns.violinplot(data=data, inner='quart', palette='muted')

# Adding title and labels
plt.title('Violin Plot', fontsize=14)
plt.xlabel('Groups', fontsize=12)
plt.ylabel('Values', fontsize=12)

# Show plot
plt.show()
