# Load dataset

# KNN:
df.isnull().sum()
df.groupby('target').size()

df.replace('?', np.nan, inplace=True)
df.head()

print(df.isnull().sum())

# to impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df))
print(df.isnull().sum())


# Load dataset
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()

# or

X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # The last column as target





## PCA

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load a sample dataset
# Replace this with your dataset
from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # Target variable

# Step 1: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
# Define the number of components you want
n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Step 3: Explained Variance Ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

# Visualize the explained variance ratio
plt.figure(figsize=(8, 5))
plt.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.7, align='center', label='Individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.title('Explained Variance Ratio by PCA Components')
plt.show()

# Step 4: Transformed Data
X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
print(X_pca_df.head())



## KNN

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load a sample dataset
# Replace this with your dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train the KNN classifier
# Specify the number of neighbors (k)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Step 5: Make predictions
y_pred = knn.predict(X_test_scaled)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)



## KMEANS

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Create or load a dataset
# Using a synthetic dataset for demonstration
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
X = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means
# Specify the number of clusters (k)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# Cluster centers and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Step 4: Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolor='k', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.legend()
plt.show()

# Step 5: Evaluate the model
# Inertia: Sum of squared distances of samples to their closest cluster center
print(f"Inertia: {kmeans.inertia_}")

# Step 6: Optional - Find the optimal number of clusters using the Elbow Method
inertia_values = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia_values, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()



## PCA and KNN

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score

# Step 1: Load a dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Apply PCA
n_components = 2  # Reduce to 2 principal components
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Step 5: Train the KNN classifier
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_pca, y_train)

# Step 6: Make predictions
y_pred_train = knn.predict(X_train_pca)
y_pred_test = knn.predict(X_test_pca)

# Step 7: Evaluate the model
# Accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

# F1-score
f1 = f1_score(y_test, y_pred_test, average='weighted')
print(f"F1 Score (weighted): {f1:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred_test)
print("\nClassification Report:")
print(class_report)

# Precision and Recall
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
print(f"Precision (weighted): {precision:.2f}")
print(f"Recall (weighted): {recall:.2f}")

# True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
tp = np.diag(conf_matrix)  # True positives for each class
fp = conf_matrix.sum(axis=0) - tp  # False positives for each class
fn = conf_matrix.sum(axis=1) - tp  # False negatives for each class
tn = conf_matrix.sum() - (tp + fp + fn)  # True negatives for each class

print("\nTP, TN, FP, FN for each class:")
for i, (tpi, tni, fpi, fni) in enumerate(zip(tp, tn, fp, fn)):
    print(f"Class {i}: TP={tpi}, TN={tni}, FP={fpi}, FN={fni}")




### PCA and KMEANS

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score

# Step 1: Load the dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # True labels (for evaluation purposes)

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
n_components = 2  # Reduce to 2 principal components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Apply K-Means Clustering
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Map clusters to true labels (optional for evaluation purposes)
# Mapping is done by finding the majority class in each cluster
from scipy.stats import mode

cluster_labels = np.zeros_like(clusters)
for i in range(k):
    mask = (clusters == i)
    cluster_labels[mask] = mode(y[mask])[0]

# Step 5: Calculate Evaluation Metrics
# Accuracy
accuracy = accuracy_score(y, cluster_labels)
print(f"Accuracy: {accuracy:.2f}")

# F1-score
f1 = f1_score(y, cluster_labels, average='weighted')
print(f"F1 Score (weighted): {f1:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y, cluster_labels)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y, cluster_labels)
print("\nClassification Report:")
print(class_report)

# Precision and Recall
precision = precision_score(y, cluster_labels, average='weighted')
recall = recall_score(y, cluster_labels, average='weighted')
print(f"Precision (weighted): {precision:.2f}")
print(f"Recall (weighted): {recall:.2f}")

# True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
tp = np.diag(conf_matrix)  # True positives for each class
fp = conf_matrix.sum(axis=0) - tp  # False positives for each class
fn = conf_matrix.sum(axis=1) - tp  # False negatives for each class
tn = conf_matrix.sum() - (tp + fp + fn)  # True negatives for each class

print("\nTP, TN, FP, FN for each class:")
for i, (tpi, tni, fpi, fni) in enumerate(zip(tp, tn, fp, fn)):
    print(f"Class {i}: TP={tpi}, TN={tni}, FP={fpi}, FN={fni}")


