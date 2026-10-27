# Machine Learning, TP2: Standardized PCA


## Objective

Implement standardized Principal Component Analysis (PCA) from scratch in Python
(NumPy and Pandas), then compare the results with the PCA implementation of
Scikit-learn.

## Dataset

Grades of six students (S1 to S6) in four subjects: Mathematics, Statistics,
Programming and Algorithms.


## Method

### Part A: PCA from scratch
1. Load the data into a Pandas DataFrame.
2. Display the shape and descriptive statistics.
3. Standardize each variable: `Z = (X - mean) / std` (sample std, ddof=1).
4. Compute the correlation matrix of the standardized data.
5. Compute the eigenvalues and eigenvectors with `np.linalg.eigh`.
6. Sort the eigenvalues in descending order and reorder the eigenvectors.
7. Keep the two eigenvectors with the largest eigenvalues.
8. Project the data: `scores = Z @ W`.
9. Compute the explained variance ratio and its cumulative sum.

### Part B: Scikit-learn
`PCA(n_components=2)` applied to the same standardized matrix `Z`, then
`fit_transform`, `explained_variance_ratio_` and its cumulative sum.

### Part C: Comparison
Coordinates, eigenvalues and explained variance of both implementations are
compared numerically, and the students are plotted in 2D for each one.

## Results

| Component | Eigenvalue | Explained variance | Cumulative |
|---|---|---|---|
| PC1 | 3.7397 | 93.49% | 93.49% |
| PC2 | 0.2402 | 6.01% | **99.50%** |
| PC3 | 0.0191 | 0.48% | 99.98% |
| PC4 | 0.0009 | 0.02% | 100% |

- Eigenvalues and explained variance ratios are **identical** in both implementations.
- The student coordinates are identical **up to the sign of PC1** (maximum
  difference of about 1e-15 after sign alignment).
- The first two components retain **99.50%** of the total variance.

## Interpretation

- **Sign difference:** an eigenvector `v` and `-v` describe the same direction
  and have the same eigenvalue. NumPy (`eigh`) and Scikit-learn (SVD) use
  different sign conventions, so PC1 comes out with opposite signs. The plot is
  mirrored along PC1, but distances between students, eigenvalues and explained
  variance are unchanged.
- **PC1 (93.49%)** is a general-performance axis: the four subjects have almost
  the same weight (about 0.5 in absolute value). S4 and S2 are the strongest
  students, S3 and S5 the weakest.
- **PC2 (6.01%)** separates students by profile, mainly Statistics against
  Programming (S6 is relatively stronger in Statistics, S1 in Programming).
- All the correlations between subjects are strong (between 0.77 and 0.99),
  which is why a single component captures most of the variance.
