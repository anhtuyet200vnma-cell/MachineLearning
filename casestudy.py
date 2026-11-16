import pandas as pd
import matplotlib.pyplot as plt

from common import (
    load_data,
    split_data,
    standardize_data,
    train_logistic_regression,
    evaluate_model,
    reduce_with_umap
)


# 1. LOAD DATA
print("===== BREAST CANCER WISCONSIN =====")

X, y, target_names = load_data()

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])


# 2. SPLIT DATA 80/20
X_train, X_test, y_train, y_test = split_data(X, y)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 3. STANDARDIZE DATA
X_train_scaled, X_test_scaled, scaler = standardize_data(
    X_train, X_test
)


# 4. BASELINE MODEL: ORIGINAL 30 FEATURES
print("\n===== BASELINE MODEL =====")

baseline_model = train_logistic_regression(
    X_train_scaled, y_train
)

baseline_accuracy, baseline_predictions = evaluate_model(
    baseline_model, X_test_scaled, y_test
)

print("Original dimensions:", X_train_scaled.shape[1])
print("Baseline accuracy:", round(baseline_accuracy, 4))


# 5. UMAP FEATURE REDUCTION
dimensions = [2, 5, 10]
results = []

for n in dimensions:

    print("\n===== UMAP", n, "DIMENSIONS =====")

    X_train_umap, X_test_umap, reducer = reduce_with_umap(
        X_train_scaled,
        X_test_scaled,
        n_components=n
    )

    model = train_logistic_regression(
        X_train_umap, y_train
    )

    accuracy, predictions = evaluate_model(
        model, X_test_umap, y_test
    )

    results.append({
        "Method": "UMAP",
        "Dimensions": n,
        "Accuracy": accuracy
    })

    print("Reduced dimensions:", n)
    print("Test accuracy:", round(accuracy, 4))


# 6. MODEL COMPARISON
comparison = pd.DataFrame([
    {
        "Method": "Original",
        "Dimensions": 30,
        "Accuracy": baseline_accuracy
    },
    *results
])

print("\n===== FINAL COMPARISON =====")
print(comparison.to_string(index=False))

comparison.to_csv(
    "umap_results.csv",
    index=False
)


# 7. ACCURACY VISUALIZATION
labels = [
    "Original 30",
    "UMAP 2",
    "UMAP 5",
    "UMAP 10"
]

accuracies = [
    baseline_accuracy,
    results[0]["Accuracy"],
    results[1]["Accuracy"],
    results[2]["Accuracy"]
]

plt.figure(figsize=(9, 5))

bars = plt.bar(labels, accuracies)

plt.title("Logistic Regression: Original vs UMAP")
plt.xlabel("Feature dimensions")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1.05)
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig("umap_accuracy_comparison.png", dpi=300)
plt.show()

print("\nExperiment completed!")