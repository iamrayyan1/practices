import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK datasets are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Example corpus
corpus = [
    "Natural Language Processing is fascinating.",
    "Machine learning and deep learning are part of AI.",
    "Text preprocessing improves NLP models."
]


# Step 1: Tokenization, Stop Words Removal, Stemming/Lemmatization
def preprocess_text(text, method='lemmatize'):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    # Stemming or Lemmatization
    if method == 'stem':
        stemmer = PorterStemmer()
        processed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    elif method == 'lemmatize':
        lemmatizer = WordNetLemmatizer()
        processed_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    else:
        raise ValueError("Method should be 'stem' or 'lemmatize'")

    return ' '.join(processed_tokens)


# Preprocess the entire corpus
method = 'lemmatize'  # Change to 'stem' for stemming
processed_corpus = [preprocess_text(text, method=method) for text in corpus]

# Step 2: TF-IDF Calculation
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_corpus)

# Display TF-IDF scores
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())
print("\nFeature Names:")
print(tfidf_vectorizer.get_feature_names_out())






#######################3


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Ensure NLTK datasets are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Example corpus
corpus = [
    "Natural Language Processing is fascinating.",
    "Machine learning and deep learning are part of AI.",
    "Text preprocessing improves NLP models."
]


# Step 1: Tokenization, Stop Words Removal, Stemming/Lemmatization
def preprocess_text(text, method='lemmatize'):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    # Stemming or Lemmatization
    if method == 'stem':
        stemmer = PorterStemmer()
        processed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    elif method == 'lemmatize':
        lemmatizer = WordNetLemmatizer()
        processed_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    else:
        raise ValueError("Method should be 'stem' or 'lemmatize'")

    return ' '.join(processed_tokens)


# Preprocess the entire corpus
method = 'lemmatize'  # Change to 'stem' for stemming
processed_corpus = [preprocess_text(text, method=method) for text in corpus]

# Step 2: TF-IDF Calculation
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_corpus)

# Step 3: Count Vectorizer Calculation
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(processed_corpus)

# Display Results
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())
print("\nTF-IDF Feature Names:")
print(tfidf_vectorizer.get_feature_names_out())

print("\nCount Vectorizer Matrix:")
print(count_matrix.toarray())
print("\nCount Vectorizer Feature Names:")
print(count_vectorizer.get_feature_names_out())









# features extractoion
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

# Ensure NLTK datasets are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Example corpus
corpus = [
    "Natural Language Processing is fascinating.",
    "Machine learning and deep learning are part of AI.",
    "Text preprocessing improves NLP models."
]


# Step 1: Tokenization, Stop Words Removal, Stemming/Lemmatization
def preprocess_text(text, method='lemmatize'):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    # Stemming or Lemmatization
    if method == 'stem':
        stemmer = PorterStemmer()
        processed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    elif method == 'lemmatize':
        lemmatizer = WordNetLemmatizer()
        processed_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    else:
        raise ValueError("Method should be 'stem' or 'lemmatize'")

    return ' '.join(processed_tokens)


# Preprocess the entire corpus
method = 'lemmatize'  # Change to 'stem' for stemming
processed_corpus = [preprocess_text(text, method=method) for text in corpus]

# Step 2: TF-IDF Calculation
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_corpus)

# Step 3: Count Vectorizer Calculation
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(processed_corpus)


# Step 4: Feature Extraction
def extract_top_features(matrix, feature_names, n=3):
    """
    Extracts the top `n` features from a given feature matrix.
    """
    top_features_per_document = []
    for doc_idx in range(matrix.shape[0]):
        row = matrix[doc_idx].toarray().flatten()
        top_indices = row.argsort()[-n:][::-1]  # Indices of top `n` features
        top_features = [(feature_names[idx], row[idx]) for idx in top_indices if row[idx] > 0]
        top_features_per_document.append(top_features)
    return top_features_per_document


# Get feature names
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
count_feature_names = count_vectorizer.get_feature_names_out()

# Extract top features
n_top_features = 3
top_tfidf_features = extract_top_features(tfidf_matrix, tfidf_feature_names, n=n_top_features)
top_count_features = extract_top_features(count_matrix, count_feature_names, n=n_top_features)

# Display Results
print("Processed Corpus:")
print(processed_corpus)

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())
print("\nTF-IDF Top Features:")
for doc_idx, features in enumerate(top_tfidf_features):
    print(f"Document {doc_idx + 1}: {features}")

print("\nCount Vectorizer Matrix:")
print(count_matrix.toarray())
print("\nCount Vectorizer Top Features:")
for doc_idx, features in enumerate(top_count_features):
    print(f"Document {doc_idx + 1}: {features}")

