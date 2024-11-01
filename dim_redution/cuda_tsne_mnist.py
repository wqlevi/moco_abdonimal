"""
run:
    python -m cuda_tsne_mnist
"""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tsnecuda import TSNE
import numpy as np
from utils.plot_utils import plot_helper

if __name__ == '__main__':
    digits = fetch_openml('mnist_784')
    data = digits.data
    print("Data has %d samples with %d features"%data.shape)

    x_train, x_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.1, shuffle=False
    )
    x_train = x_train.to_numpy(dtype=np.float32)

    tsne = TSNE(perplexity=30,init='random',metric='euclidean',verbose=1
                )

    embedding_train = tsne.fit_transform(x_train)
    plot_helper(embedding_train, y_train, title="cuda-tsne")
