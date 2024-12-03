import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = np.random.rand(10, 5)

# Box plot
plt.figure(figsize=(8, 5))
plt.boxplot(data, patch_artist=True, notch=True, vert=False, widths=0.7, boxprops=dict(facecolor='lightblue', color='blue'))

# Adding title and labels
plt.title('Box Plot', fontsize=14)
plt.xlabel('Values', fontsize=12)

# Show plot
plt.show()
