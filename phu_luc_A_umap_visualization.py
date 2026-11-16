import matplotlib.pyplot as plt

from common import (
    load_data,
    split_data,
    standardize_data,
    reduce_with_umap
)


# Load data
X, y, target_names = load_data()

# Split data
X_train, X_test, y_train, y_test = split_data(X, y)

# Standardize
X_train_scaled, X_test_scaled, scaler = standardize_data(
    X_train, X_test
)

# UMAP 2D
X_umap, X_test_umap, reducer = reduce_with_umap(
    X_train_scaled,
    X_test_scaled,
    n_components=2
)

# Visualize training data
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=y_train,
    cmap="coolwarm",
    s=20,
    alpha=0.8
)

plt.title("UMAP 2D Visualization - Breast Cancer")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

plt.colorbar(scatter, label="Class")
plt.tight_layout()

plt.savefig("umap_2d_visualization.png", dpi=300)
plt.show()