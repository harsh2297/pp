import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/harsh/Desktop/QUINT_F/csv_files/text_processed.csv", header  = None)
print data[:5]
data.columns = ['id','category','text']
del data['id']
#print data[:5]
labels = data['category']
X = data['text']

#train_x, valid_x = model_selection.train_test_split(df)


''''# create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(train_x)

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)'''

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)


''''# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000, stop_words='english')
tfidf_vect.fit(df)
xtrain_tfidf =  tfidf_vect.transform(train_x)


# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(df)
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)

# characters level tf-idf
#tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
#tfidf_vect_ngram_chars.fit(df)
#xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)


true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
#m1 = model.fit(xtrain_count)
m2 = model.fit(xtrain_tfidf)
m3 = model.fit(xtrain_tfidf_ngram)
#m4 = model.fit(xtrain_tfidf_ngram_chars)'''

# k means determine k
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

### One more method for

for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(X)
    label = kmeans.labels_
    sil_coeff = silhouette_score(X, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
##dictionary = corpora.Dictionary(texts)
# corpus = [dictionary.doc2bow(text) for text in texts]
# df9['Corpus'] = pd.Series(corpus, index = df9.index)

##### Inertia error method
cluster_range = range(1, 30)
cluster_errors = []
for num_clusters in cluster_range:
    clusters = KMeans(num_clusters)
    clusters.fit(X)
    cluster_errors.append(clusters.inertia_)
clusters_df = pd.DataFrame({"num_clusters": cluster_range, "cluster_errors": cluster_errors})

plt.figure(figsize=(12, 6))
plt.plot(clusters_df.num_clusters, clusters_df.cluster_errors, marker="o")

'''true_k = 5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
m1 = model.fit(df)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

Y = vectorizer.transform(valid_x)
prediction1 = m1.predict(Y)
#prediction2 = m1.predict(Y)
#prediction3 = m1.predict(Y)
#prediction4 = m1.predict(Y)

k = pd.DataFrame(prediction1)
k[]
#print(prediction2)
#print(prediction3)
#print(prediction4)
prediction1.to_csv('tfidf_cluster.csv')
#prediction3.to_csv('tfidf_ngrams_cluster.csv')
'''


