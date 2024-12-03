import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [25, 40, 35, 50]

# Create a DataFrame for the data
import pandas as pd
data = pd.DataFrame({
    'Category': categories,
    'Value': values
})

# Bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x='Category', y='Value', data=data, palette='viridis')

# Adding title and labels
plt.title('Bar Plot', fontsize=14)
plt.xlabel('Categories', fontsize=12)
plt.ylabel('Values', fontsize=12)

# Show plot
plt.show()
