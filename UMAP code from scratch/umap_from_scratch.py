"""
UMAP + Logistic Regression FROM SCRATCH
- UMAP: NumPy only (KNN, local fuzzy memberships, fuzzy graph, SGD + negative sampling)
- Logistic Regression: NumPy only (binary cross-entropy + gradient descent)
- StandardScaler, stratified split, PCA: implemented manually
- scikit-learn is used ONLY to load the public Breast Cancer Wisconsin dataset.
No sklearn estimator/model is used.
Run: python umap_from_scratch.py
"""

import numpy as np
import csv
from pathlib import Path


# --------------------------- Reproducibility ---------------------------

SEED = 42
np.random.seed(SEED)


# --------------------------- Data utilities ----------------------------

def stratified_split(X, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def standardize_transform(X, mean, std):
    return (X - mean) / std


# ------------------------- Metrics (NumPy) -----------------------------

def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def classification_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return accuracy, precision, recall, f1, np.array([[tn, fp], [fn, tp]])


# ---------------- Logistic Regression from scratch -------------------

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.05, epochs=3000, l2=0.01):
        self.lr = learning_rate
        self.epochs = epochs
        self.l2 = l2

    def fit(self, X, y):
        Xb = np.column_stack([np.ones(len(X)), X])
        self.w = np.zeros(Xb.shape[1])
        for _ in range(self.epochs):
            p = sigmoid(Xb @ self.w)
            grad = (Xb.T @ (p - y)) / len(y)
            grad[1:] += self.l2 * self.w[1:]
            self.w -= self.lr * grad
        return self

    def predict(self, X):
        Xb = np.column_stack([np.ones(len(X)), X])
        return (sigmoid(Xb @ self.w) >= 0.5).astype(int)


# --------------------------- PCA from scratch --------------------------

def pca_fit_transform(X_train, X_test, n_components):
    mean = X_train.mean(axis=0)
    centered_train = X_train - mean
    centered_test = X_test - mean
    cov = centered_train.T @ centered_train / (len(X_train) - 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1][:n_components]
    components = vectors[:, order]
    return centered_train @ components, centered_test @ components


# ----------------------------- UMAP ------------------------------------

def pairwise_distances(X):
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2 * (X @ X.T), 0.0)
    return np.sqrt(d2)


def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0):
    """Estimate rho_i and sigma_i so sum_j exp(-(d_ij-rho_i)/sigma_i) ~= log2(k)."""
    n = distances.shape[0]
    target = np.log2(k)
    rho = np.zeros(n)
    sigma = np.ones(n)

    for i in range(n):
        positive = distances[i][distances[i] > 0]
        if len(positive):
            # local_connectivity=1: use nearest positive neighbor as rho
            rho[i] = positive[min(int(local_connectivity) - 1, len(positive) - 1)]
        lo, hi = 0.0, np.inf
        mid = 1.0
        for _ in range(n_iter):
            membership_sum = np.sum(
                np.exp(-np.maximum(distances[i] - rho[i], 0.0) / max(mid, 1e-12))
            ) - 1.0  # exclude self
            if abs(membership_sum - target) < 1e-5:
                break
            if membership_sum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                mid = mid * 2.0 if np.isinf(hi) else (lo + hi) / 2.0
        sigma[i] = max(mid, 1e-3)
    return rho, sigma


class UMAPScratch:
    def __init__(self, n_components=2, n_neighbors=15, n_epochs=180,
                 learning_rate=0.35, negative_sample_rate=5, seed=42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_epochs = n_epochs
        self.lr = learning_rate
        self.negative_sample_rate = negative_sample_rate
        self.seed = seed
        # Standard UMAP curve approximation used by the common min_dist=0.1 setting
        self.a = 1.576943460
        self.b = 0.895060879

    def _fuzzy_graph(self, X):
        n = len(X)
        D = pairwise_distances(X)
        k = min(self.n_neighbors, n - 1)
        knn_idx = np.argsort(D, axis=1)[:, 1:k + 1]
        knn_dist = np.take_along_axis(D, knn_idx, axis=1)

        rho = np.zeros(n)
        sigma = np.ones(n)
        for i in range(n):
            positive = knn_dist[i][knn_dist[i] > 0]
            if len(positive):
                rho[i] = positive[0]
            target = np.log2(k)
            lo, hi, mid = 0.0, np.inf, 1.0
            for _ in range(50):
                s = np.sum(np.exp(-np.maximum(knn_dist[i] - rho[i], 0) / max(mid, 1e-12)))
                if abs(s - target) < 1e-5:
                    break
                if s > target:
                    hi = mid
                    mid = (lo + hi) / 2
                else:
                    lo = mid
                    mid = mid * 2 if np.isinf(hi) else (lo + hi) / 2
            sigma[i] = max(mid, 1e-3)

        directed = np.zeros((n, n), dtype=float)
        rows = np.arange(n)[:, None]
        directed[rows, knn_idx] = np.exp(
            -np.maximum(knn_dist - rho[:, None], 0) / sigma[:, None]
        )
        # Fuzzy union: p_ij + p_ji - p_ij*p_ji
        graph = directed + directed.T - directed * directed.T
        np.fill_diagonal(graph, 0.0)
        return graph

    def _spectral_initialization(self, graph, rng):
        degree = graph.sum(axis=1)
        inv = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
        lap = np.eye(len(graph)) - (inv[:, None] * graph * inv[None, :])
        vals, vecs = np.linalg.eigh(lap)
        embedding = vecs[:, 1:self.n_components + 1]
        embedding -= embedding.mean(axis=0, keepdims=True)
        scale = np.std(embedding)
        if scale > 0:
            embedding /= scale
        embedding += rng.normal(0, 1e-4, embedding.shape)
        return embedding * 4.0

    def fit_transform(self, X):
        rng = np.random.default_rng(self.seed)
        self.X_train = np.asarray(X, dtype=float)
        graph = self._fuzzy_graph(self.X_train)
        self.embedding_ = self._spectral_initialization(graph, rng)

        edge_i, edge_j = np.where(np.triu(graph, 1) > 0)
        weights = graph[edge_i, edge_j]
        # Sample edges in proportion to fuzzy membership strength
        n_edges = max(1, int(np.sum(weights) * 2.0))
        probs = weights / weights.sum()
        sample_i = rng.choice(edge_i, size=n_edges, replace=True, p=probs)
        sample_j = rng.choice(edge_j, size=n_edges, replace=True, p=probs)

        Y = self.embedding_.copy()
        n = len(Y)
        for epoch in range(self.n_epochs):
            order = rng.permutation(n_edges)
            lr = self.lr * (1.0 - epoch / self.n_epochs)
            for start in range(0, n_edges, 256):
                batch = order[start:start + 256]
                ii, jj = sample_i[batch], sample_j[batch]
                diff = Y[ii] - Y[jj]
                dist2 = np.sum(diff * diff, axis=1) + 1e-8

                # Attractive gradient from -log(q_ij)
                coeff = (2.0 * self.a * self.b *
                         np.power(dist2, self.b - 1.0) /
                         (1.0 + self.a * np.power(dist2, self.b)))
                grad = coeff[:, None] * diff
                np.add.at(Y, ii, -lr * grad)
                np.add.at(Y, jj, lr * grad)

                # Negative sampling approximates the repulsive term
                for _ in range(self.negative_sample_rate):
                    kk = rng.integers(0, n, size=len(ii))
                    neg_diff = Y[ii] - Y[kk]
                    neg_d2 = np.sum(neg_diff * neg_diff, axis=1) + 1e-8
                    rep = (2.0 * self.b /
                           ((0.001 + neg_d2) *
                            (1.0 + self.a * np.power(neg_d2, self.b))))
                    rep_grad = rep[:, None] * neg_diff
                    np.add.at(Y, ii, lr * rep_grad)
                    np.add.at(Y, kk, -lr * rep_grad)

            # Keep the embedding numerically centered and bounded
            Y -= Y.mean(axis=0, keepdims=True)
            Y = np.clip(Y, -20, 20)

        self.embedding_ = Y
        return Y

    def transform(self, X_new):
        """Out-of-sample placement: weighted barycentric interpolation from train neighbors."""
        X_new = np.asarray(X_new, dtype=float)
        result = np.zeros((len(X_new), self.n_components))
        k = min(self.n_neighbors, len(self.X_train))
        for i, x in enumerate(X_new):
            d = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            idx = np.argsort(d)[:k]
            positive = d[idx][d[idx] > 0]
            rho = positive[0] if len(positive) else 0.0
            weights = np.exp(-np.maximum(d[idx] - rho, 0.0) / max(np.mean(d[idx]) * 0.5, 1e-8))
            weights /= max(weights.sum(), 1e-12)
            result[i] = weights @ self.embedding_[idx]
        return result


# ------------------------------- Main ----------------------------------

def evaluate(name, X_train, X_test, y_train, y_test):
    model = LogisticRegressionScratch(learning_rate=0.05, epochs=3500, l2=0.01)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc, prec, rec, f1, cm = classification_metrics(y_test, pred)
    print(f"{name:<24} Accuracy={acc:.4f} ({acc*100:.2f}%) | "
          f"Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")
    print(f"  Confusion matrix [[TN, FP], [FN, TP]] = {cm.tolist()}")
    return acc


def main():
    # Read local CSV using Python's standard library (no scikit-learn dependency).
    csv_file = Path(__file__).with_name("breast_cancer_wisconsin.csv")
    with csv_file.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    values = np.asarray([[float(v) for v in row] for row in rows[1:]], dtype=float)
    X = values[:, :-1]
    # Dataset target: 0=malignant, 1=benign. Convert to malignant=1.
    y = (values[:, -1] == 0).astype(int)

    X_train, X_test, y_train, y_test = stratified_split(X, y, 0.2, SEED)
    X_train_s, mean, std = standardize_fit(X_train)
    X_test_s = standardize_transform(X_test, mean, std)

    print("=" * 78)
    print("BREAST CANCER WISCONSIN — IMPLEMENTATION FROM SCRATCH")
    print("Dataset: local breast_cancer_wisconsin.csv (read with Python csv)")
    print("All preprocessing, UMAP, PCA, Logistic Regression and metrics: NumPy")
    print("=" * 78)
    print(f"Total samples: {len(X)} | Features: {X.shape[1]}")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Train classes (benign=0, malignant=1): {np.bincount(y_train).tolist()}")
    print(f"Test classes  (benign=0, malignant=1): {np.bincount(y_test).tolist()}")
    print("\n--- Logistic Regression evaluation ---")

    scores = {}
    scores["Original 30 features"] = evaluate(
        "Original 30 features", X_train_s, X_test_s, y_train, y_test
    )

    for dim in (2, 5, 10):
        print(f"\nTraining UMAP from scratch: n_components={dim} ...")
        umap = UMAPScratch(n_components=dim, n_neighbors=15,
                           n_epochs=180, learning_rate=0.35, seed=SEED)
        Z_train = umap.fit_transform(X_train_s)
        Z_test = umap.transform(X_test_s)
        scores[f"UMAP {dim}D"] = evaluate(
            f"UMAP {dim}D", Z_train, Z_test, y_train, y_test
        )

    print("\n--- PCA baseline (implemented with NumPy) ---")
    for dim in (2, 5, 10):
        Z_train, Z_test = pca_fit_transform(X_train_s, X_test_s, dim)
        scores[f"PCA {dim}D"] = evaluate(
            f"PCA {dim}D", Z_train, Z_test, y_train, y_test
        )

    print("\n--- Summary (Accuracy) ---")
    for name, score in scores.items():
        print(f"{name:<24}: {score*100:6.2f}%")
    print("\nNote: UMAP is stochastic and this is a compact educational implementation.")
    print("The test set is not used to fit the scaler, UMAP graph, or classifier.")


if __name__ == "__main__":
    main()
