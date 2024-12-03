import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = np.random.rand(10, 5)

# Box plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, palette='muted', fliersize=8, linewidth=2)

# Adding title and labels
plt.title('Box Plot', fontsize=14)
plt.xlabel('Groups', fontsize=12)
plt.ylabel('Values', fontsize=12)

# Show plot
plt.show()
