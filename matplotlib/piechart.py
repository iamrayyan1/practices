import matplotlib.pyplot as plt

# Data
labels = ['A', 'B', 'C', 'D']
sizes = [25, 30, 20, 25]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

# Pie chart
plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')

# Adding title
plt.title('Pie Chart', fontsize=14)

# Show plot
plt.show()
