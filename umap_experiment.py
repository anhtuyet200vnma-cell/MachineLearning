import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd

from common import (
    evaluate_model,
    load_data,
    reduce_with_umap,
    split_data,
    standardize_data,
    train_logistic_regression,
)


def main():
    X, y, _ = load_data()
    print("BREAST CANCER WISCONSIN")
    print("Samples:", X.shape[0])
    print("Features:", X.shape[1])

    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    X_train_scaled, X_test_scaled, _ = standardize_data(X_train, X_test)

    baseline = train_logistic_regression(X_train_scaled, y_train)
    baseline_accuracy, _ = evaluate_model(baseline, X_test_scaled, y_test)

    results = [
        {
            "Method": "Original",
            "Dimensions": X.shape[1],
            "Accuracy": baseline_accuracy,
        }
    ]

    for dimensions in (2, 5, 10):
        X_train_umap, X_test_umap, _ = reduce_with_umap(
            X_train_scaled,
            X_test_scaled,
            n_components=dimensions,
        )
        classifier = train_logistic_regression(X_train_umap, y_train)
        accuracy, _ = evaluate_model(classifier, X_test_umap, y_test)
        results.append(
            {
                "Method": "UMAP",
                "Dimensions": dimensions,
                "Accuracy": accuracy,
            }
        )

    comparison = pd.DataFrame(results)
    print("\nCOMPARISON")
    print(comparison.to_string(index=False, formatters={"Accuracy": "{:.4f}".format}))
    comparison.to_csv("umap_results.csv", index=False)

    labels = ["Original (30)", "UMAP (2)", "UMAP (5)", "UMAP (10)"]
    accuracies = comparison["Accuracy"].tolist()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, accuracies)
    plt.title("Comparison of Classification Accuracy")
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.01,
            f"{accuracy:.3f}",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig("umap_accuracy_comparison.png", dpi=300)
    plt.close()
    print("\nSaved umap_results.csv and umap_accuracy_comparison.png")


if __name__ == "__main__":
    main()
