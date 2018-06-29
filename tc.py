# -*- coding: utf-8 -*-

import collections
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint


data = pd.read_csv("/Users/harsh/Desktop/QUINT_F/csv_files/text_processed.csv", header  = None)
data.columns = ['id','category','text']
df = data['text']
def get_cluster_kmeans(tfidf_matrix, num_clusters):
    km = KMeans(n_clusters = num_clusters)
    km.fit(tfidf_matrix)
    cluster_list = km.labels_.tolist()
    return cluster_list

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2)
p = tfidf_vectorizer.fit_transform(df) #fit the vectorizer to synopses

cl = get_cluster_kmeans(p,5)
cldf = pd.DataFrame(cl)
cldf['text'] = df
cldf.to_csv("cldf.csv")














































'''df = data['text']
vectorizer = TfidfVectorizer(max_df=0.5,min_df=0.1,lowercase=True)

tfidf_model = vectorizer.fit_transform(df)

km_model = KMeans(n_clusters=5)
km_model.fit(tfidf_model)
clustering = collections.defaultdict(list)

for idx, label in enumerate(km_model.labels_):
    clustering[label].append(idx)
    print clustering

if __name__ == "__main__":
    articles = df.to_list()
    clusters = cluster_texts(articles, 5)
    pprint(dict(clusters))
'''