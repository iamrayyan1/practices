import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('kmeans.csv')


X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # The last column as target

# n features 2

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.cluster import KMeans

# manual
wcss = []
for k in range(1,11):
    kmeans=KMeans(n_clusters=k, init='k-means++')
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)

print(wcss)

plt.plot(range(1,11), wcss)
plt.xticks(range(1,11))
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

y_labels = kmeans.fit_predict(X_train)
plt.scatter(X_train[:,0], X_train[:,1], c=y_labels)
plt.show()

y_test_labels=kmeans.predict(X_test)
plt.scatter(X_train[:,0], X_train[:,1], c=y_test_labels)
plt.show()



# automatic through knee locator
from kneed import KneeLocator
kl=KneeLocator(range(1,11), wcss, curve='convex', direction='decreasing')
kl.elbow


# performance matrix
# Silhoutte score

from sklearn.metrics import silhouette_score
silhouette_coefficients=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X_train)
    score = silhouette_score(X_train, kmeans.labels_)
    silhouette_coefficients.append(score)


silhouette_coefficients

plt.plot(range(1,11), silhouette_coefficients)
plt.xticks(range(1,11))
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette coefficient')
plt.show()