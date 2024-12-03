import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Line plot
plt.figure(figsize=(8, 5))
sns.lineplot(x=x, y=y, label='Sine Wave', color='b')

# Adding title and labels
plt.title('Line Plot', fontsize=14)
plt.xlabel('X Axis', fontsize=12)
plt.ylabel('Y Axis', fontsize=12)

# Show legend
plt.legend()

# Display grid
plt.grid(True)

# Show plot
plt.show()
