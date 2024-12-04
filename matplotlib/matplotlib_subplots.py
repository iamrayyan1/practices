
import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Basic subplots
fig, ax = plt.subplots(2, 1, figsize=(8, 6))  # 2 rows, 1 column

# First subplot
ax[0].plot(x, y1, 'r')
ax[0].set_title('Sine Wave')
ax[0].set_xlabel('X-axis')
ax[0].set_ylabel('Y-axis')

# Second subplot
ax[1].plot(x, y2, 'b')
ax[1].set_title('Cosine Wave')
ax[1].set_xlabel('X-axis')
ax[1].set_ylabel('Y-axis')

# Adjust layout and display
plt.tight_layout()
plt.show()

# Complex subplot layout
fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid
data = [y1, y2, -y1, -y2]
titles = ['Sine', 'Cosine', 'Negative Sine', 'Negative Cosine']
colors = ['red', 'blue', 'green', 'orange']

for i, ax in enumerate(axs.flat):  # Flatten the 2D array of axes
    ax.plot(x, data[i], color=colors[i])
    ax.set_title(titles[i])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

plt.tight_layout()
plt.show()
