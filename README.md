# Clustering-News-Headlines
The project groups scrapped News headlines using NLTK, KMeans and Ward Hierarchical Method.


Table of Contents 
---------------------------
scrapping.py

	Scrap/Extract data using BeautifulSoup
	
preprocessing.py

	Clean data to remove punctuation, numbers, white spaces and convert the text to lower case
	
k_means.py

	Tokenize and Stem Data using NLTK
	Convert words to Vector Space using TFIDF matrix
	Calculate Cosine Similarity and generate the distance matrix
	Generate Clusters using KMeans clustering algorithm
	Dimensionality reduction using MDS
	Visualization of clusters using matplotlib
	
hierarchical.py

	Tokenize and Stem Data using NLTK
	Convert words to Vector Space using TFIDF matrix
	Calculate Cosine Similarity and generate the distance matrix
	Hierarchical clustering using Ward Method
	Visualization of clusters using matplotlib


The different processes and algorithms used are explained in [Clustering News Headlines](https://github.com/maneeshavinayak/Clustering-News-Headlines/blob/master/doc/Clustering%20News%20Headlines.docx)

Project Requirements
----------------------------

python 3
pip install requirements.txt

Run below from python command line 

	import nltk
	nltk.download('all')


