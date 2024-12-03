import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [10, 15, 8, 12]
errors = [1, 2, 1.5, 1]  # Error values for each bar

# Creating the bar plot
plt.figure(figsize=(10, 6))  # Set the figure size
bars = plt.bar(categories, values, yerr=errors, capsize=5, color=['blue', 'orange', 'green', 'purple'])

# Title and labels
plt.title("Bar Plot with Annotations and Error Bars", fontsize=16, fontweight='bold')
plt.xlabel("Categories", fontsize=14)
plt.ylabel("Values", fontsize=14)

# Adding grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding data labels (annotations)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height}', 
             ha='center', va='bottom', fontsize=12)

# Customizing the axes
plt.xticks(fontsize=12)
plt.yticks(np.arange(0, 20, 2), fontsize=12)

# Adding legend
plt.legend(['Values'], loc='upper left')

# Display the plot
plt.tight_layout()  # Adjust layout for better fit
plt.show()
