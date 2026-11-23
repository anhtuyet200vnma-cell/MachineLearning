import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

class DecisionTreeFromScratch:
    def __init__(self, max_depth=2, ccp_alpha=0.02):
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha

    @staticmethod
    def _gini(y):
        # Gini = 1 - tổng bình phương tỷ lệ các lớp
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return float(1 - np.sum(p ** 2))

    def _build_tree(self, X, y, depth):
        labels, counts = np.unique(y, return_counts=True)

        node = {
            "label": int(labels[np.argmax(counts)]),
            "n": len(y),
            "gini": self._gini(y),
            "left": None,
            "right": None
        }

        if depth >= self.max_depth or len(labels) == 1:
            return node

        best_score = node["gini"]
        best_split = None

        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature]).astype(float)
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                mask = X[:, feature] <= threshold
                n_left = int(mask.sum())
                n_right = len(y) - n_left

                if n_left == 0 or n_right == 0:
                    continue

                score = (
                    n_left * self._gini(y[mask])
                    + n_right * self._gini(y[~mask])
                ) / len(y)

                if score < best_score - 1e-12:
                    best_score = score
                    best_split = (
                        feature,
                        float(threshold),
                        mask
                    )

        if best_split is None:
            return node

        feature, threshold, mask = best_split

        node["feature"] = feature
        node["threshold"] = threshold

        node["left"] = self._build_tree(
            X[mask], y[mask], depth + 1
        )
        node["right"] = self._build_tree(
            X[~mask], y[~mask], depth + 1
        )

        return node

    def _prune(self, node):
     
        leaf_cost = (
            node["n"] / self.n_train * node["gini"]
            + self.ccp_alpha
        )

        if node["left"] is None:
            return leaf_cost

        keep_cost = (
            self._prune(node["left"])
            + self._prune(node["right"])
        )

        if leaf_cost <= keep_cost + 1e-12:
            node["left"] = None
            node["right"] = None
            return leaf_cost

        return keep_cost

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if (
            X.ndim != 2
            or y.ndim != 1
            or len(y) == 0
            or len(X) != len(y)
            or X.shape[1] == 0
            or not np.isfinite(X).all()
        ):
            raise ValueError("Dữ liệu train không hợp lệ.")

        self.n_train = len(y)
        self.n_features = X.shape[1]

        self.root = self._build_tree(X, y, depth=0)

        if self.ccp_alpha > 0:
            self._prune(self.root)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)

        if (
            X.ndim != 2
            or X.shape[1] != self.n_features
            or not np.isfinite(X).all()
        ):
            raise ValueError("Dữ liệu dự đoán không hợp lệ.")

        predictions = []

        for sample in X:
            node = self.root

            while node["left"] is not None:
                if sample[node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]

            predictions.append(node["label"])

        return np.asarray(predictions)

    def print_tree(self, feature_names):
        def show(node, indent=""):
            if node["left"] is None:
                print(f'{indent}|--- class: {node["label"]}')
                return

            name = feature_names[node["feature"]]
            threshold = node["threshold"]

            print(f"{indent}|--- {name} <= {threshold:.2f}")
            show(node["left"], indent + "|   ")

            print(f"{indent}|--- {name} >  {threshold:.2f}")
            show(node["right"], indent + "|   ")

        show(self.root)

if __name__ == "__main__":
    wine = load_wine()

    X = wine.data[:, [9]]
    y = wine.target + 1

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeFromScratch(
        max_depth=2,
        ccp_alpha=0.02
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    so_dung = int(np.sum(y_pred == y_test))

    print("Số mẫu train:", len(y_train))
    print("Số mẫu test:", len(y_test))
    print("Dự đoán đúng:", so_dung)
    print("Dự đoán sai:", len(y_test) - so_dung)
    print(f"Accuracy: {so_dung / len(y_test):.2%}")

    print("\nCÂY SAU CẮT TỈA")
    model.print_tree(["Cuong_do_mau"])

    duong_dan = Path(__file__).resolve().parent / "cay_wine.json"

    with duong_dan.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_features": model.n_features,
                "root": model.root
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nĐã lưu mô hình:", duong_dan)