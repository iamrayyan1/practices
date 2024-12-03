import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import silhouette_score
import nltk

# Download necessary resources from nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Example email dataset
data = {
    'email_id': [1, 2, 3],
    'text': [
        "Win a $1000 prize now! Click here for more details.",
        "Meeting scheduled at 3 PM tomorrow. Bring the documents.",
        "Congratulations! You've won a free vacation to the Bahamas. Act fast!"
    ]
}

# Load data into DataFrame
emails_df = pd.DataFrame(data)

# Preprocessing Functions

# 1. Tokenization
def tokenize_text(text):
    return word_tokenize(text)

# 2. Remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# 3. Stop word removal
stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in stop_words]

# 4. Stemming
stemmer = PorterStemmer()
def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]

# 5. Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

# Apply preprocessing to emails
def preprocess_text(text, lemmatize=True):
    # Tokenize
    tokens = tokenize_text(text)
    # Remove punctuation
    text_no_punct = remove_punctuation(' '.join(tokens))
    # Tokenize again after punctuation removal
    tokens = tokenize_text(text_no_punct)
    # Remove stopwords
    tokens_no_stop = remove_stopwords(tokens)
    # Apply stemming or lemmatization
    if lemmatize:
        processed_tokens = lemmatize_tokens(tokens_no_stop)
    else:
        processed_tokens = stem_tokens(tokens_no_stop)
    return ' '.join(processed_tokens)

# Process all emails
emails_df['processed_text'] = emails_df['text'].apply(lambda x: preprocess_text(x, lemmatize=True))

# Vectorization

# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform(emails_df['processed_text'])

# Dimensionality Reduction using PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
reduced_features = pca.fit_transform(tfidf_vectors.toarray())

# KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42, init='k-means++')
clusters = kmeans.fit_predict(reduced_features)

# Adding clustering results to the DataFrame
emails_df['cluster'] = clusters

# Evaluation using Silhouette Score
silhouette = silhouette_score(reduced_features, clusters)
print(f"Silhouette Score: {silhouette}")

# Display results
print("\nOriginal Emails:\n", emails_df['text'])
print("\nProcessed Emails:\n", emails_df['processed_text'])
print("\nCluster Assignments:\n", emails_df[['email_id', 'cluster']])
