import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.random.rand(100)
y = np.random.rand(100)

# Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='green', alpha=0.6, edgecolors='black')

# Adding title and labels
plt.title('Scatter Plot', fontsize=14)
plt.xlabel('X Axis', fontsize=12)
plt.ylabel('Y Axis', fontsize=12)

# Display grid
plt.grid(True)

# Show plot
plt.show()
