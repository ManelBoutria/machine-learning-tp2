import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = {
    "Student": ["S1", "S2", "S3", "S4", "S5", "S6"],
    "Mathematics": [12, 15, 8, 17, 10, 14],
    "Statistics": [11, 14, 9, 16, 12, 15],
    "Programming": [14, 16, 10, 18, 11, 13],
    "Algorithms": [13, 15, 9, 17, 10, 14],
}
df = pd.DataFrame(data).set_index("Student")
print(df)
print("Shape:", df.shape)
print(df.describe())
X = df.values.astype(float)
mean = X.mean(axis=0)
std = X.std(axis=0, ddof=1)
Z = (X - mean) / std
print(pd.DataFrame(Z, index=df.index, columns=df.columns).round(4))
corr = np.corrcoef(Z, rowvar=False)
print(pd.DataFrame(corr, index=df.columns, columns=df.columns).round(4))
cov = np.cov(Z, rowvar=False)      # = matrice de corrélation (données standardisées)
eigvals, eigvecs = np.linalg.eigh(cov)
print(eigvals)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]
print(eigvals)
print(eigvecs)
W = eigvecs[:, :2]
print(W)
scores_scratch = Z @ W
print(pd.DataFrame(scores_scratch, index=df.index, columns=["PC1", "PC2"]).round(4))
evr_scratch = eigvals / eigvals.sum()
cum_scratch = np.cumsum(evr_scratch)
print(evr_scratch)
print(cum_scratch)
pca = PCA(n_components=2)
scores_sk = pca.fit_transform(Z)    # on donne Z (déjà standardisé)
print(pd.DataFrame(scores_sk, index=df.index, columns=["PC1", "PC2"]).round(4))
evr_sk = pca.explained_variance_ratio_
cum_sk = np.cumsum(evr_sk)
print(evr_sk)
print(cum_sk)
print("Identiques ?        ", np.allclose(scores_scratch, scores_sk))
print("Identiques au signe près ?", np.allclose(np.abs(scores_scratch), np.abs(scores_sk)))
signs = np.sign(np.sum(scores_scratch * scores_sk, axis=0))
print("Signes relatifs :", signs)
print("Valeurs propres :", eigvals[:2], pca.explained_variance_)
print("Ratios          :", evr_scratch[:2], evr_sk)
print(np.allclose(eigvals[:2], pca.explained_variance_))
print(np.allclose(evr_scratch[:2], evr_sk))
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, sc, title in zip(axes, [scores_scratch, scores_sk],
                         ["PCA from scratch", "PCA scikit-learn"]):
    ax.scatter(sc[:, 0], sc[:, 1], c="#d9730d", s=70)
    for name, (x, y) in zip(df.index, sc):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6))
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({evr_scratch[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr_scratch[1]*100:.1f}%)")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
print(f"From scratch : {cum_scratch[1]*100:.2f} %")
print(f"scikit-learn : {cum_sk[1]*100:.2f} %")

