from sklearn import metrics, svm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from openTSNE import TSNE
import openTSNE
import numpy as np

from utils.plot_utils import plot_helper



digits = fetch_openml('mnist_784')
data = digits.data
print("Data has %d samples with %d features"%data.shape)

x_train, x_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.1, shuffle=False
)
x_train = x_train.to_numpy(dtype=np.float32)

print("%d training samples" % x_train.shape[0])
print("%d test samples" % x_test.shape[0])

#vectorizer = TfidfVectorizer(sublinear_tf=True)
#tfidf_features = vectorizer.fit_transform(x_train)
svd = TruncatedSVD(n_components=500, random_state=42, algorithm='arpack')
x = svd.fit_transform(x_train)
A = openTSNE.affinity.Uniform(x, method='exact', random_state=42, k_neighbors=15)
tsne = TSNE(initialization='spectral', random_state=42)
embedding_train = tsne.fit(affinities=A)
"""
tsne = TSNE(
    perplexity=300,
    initialization='pca',
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)
"""
#embedding_train = tsne.fit(x_train.to_numpy())

plot_helper(embedding_train, y_train, title="tsne_spectral")
