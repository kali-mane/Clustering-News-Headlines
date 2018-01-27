# Tokenize and Stem Data
# Convert words to Vector Space using TFIDF matrix
# Calculate Cosine Similarity and generate the distance matrix
# Uses Ward method to generate an hierarchy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import ward, dendrogram
import os


# Function to return a list of stemmed words
def tokenize_and_stem(text_file):
    # declaring stemmer and stopwords language
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text_file)
    filtered = [w for w in words if w not in stop_words]
    stems = [stemmer.stem(t) for t in filtered]
    return stems


def main():

    path = os.path.abspath(os.path.dirname(__file__))
    data = pd.read_csv(os.path.join(path, 'data\headlines_cleaned.txt'), names=['text'])

    # text data in dataframe and removing stops words
    stop_words = set(stopwords.words('english'))
    data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Using TFIDF vectorizer to convert convert words to Vector Space
    tfidf_vectorizer = TfidfVectorizer(max_features=200000,
                                       use_idf=True,
                                       stop_words='english',
                                       tokenizer=tokenize_and_stem)
    #                                   ngram_range=(1, 3))

    # Fit the vectorizer to text data
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'])

    # Calculating the distance measure derived from cosine similarity
    distance = 1 - cosine_similarity(tfidf_matrix)

    # Ward’s method produces a hierarchy of clusterings
    linkage_matrix = ward(distance)
    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = dendrogram(linkage_matrix, orientation="top", labels=data.values)
    plt.tight_layout()
    plt.title('News Headlines using Ward Hierarchical Method')
    plt.savefig(os.path.join(path, 'results\hierarchical.png'))


if __name__ == '__main__':
    main()
