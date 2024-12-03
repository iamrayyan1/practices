import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sample data - Using Seaborn's built-in Iris dataset
iris = sns.load_dataset('iris')

# Pair plot
plt.figure(figsize=(8, 8))
sns.pairplot(iris, hue='species', palette='Set1', markers=["o", "s", "D"])

# Show plot
plt.show()
