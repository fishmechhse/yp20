import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_projection(XD, y, title):
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(XD[:, 0], XD[:, 1], c=y,  alpha=0.6, s=30)
    plt.xlabel('Компонента 1')
    plt.ylabel('Компонента 2')
    plt.title(title)
    plt.colorbar(scatter, label='features')
    plt.grid(True)
    plt.show()

def make_pca(X: pd.DataFrame, y):
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   X_mean = np.mean(X.to_numpy(), axis=0)
   X_minus_mean = X.to_numpy() - X_mean

   # Step 3: Apply PCA
   pca = PCA(n_components=2)  # We want to reduce to 2 dimensions
   X_pca = pca.fit_transform(X_scaled)

   plot_projection(X_pca, y, 'v5_v6 features')

