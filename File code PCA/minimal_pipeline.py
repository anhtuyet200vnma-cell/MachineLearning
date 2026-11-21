"""Minimal leakage-free PCA pipeline (Listing 3.1 of the paper)."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score

data = load_breast_cancer()                     # 0 = malignant, 1 = benign
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.20, random_state=42, stratify=data.target)

model = Pipeline([
    ("scale", StandardScaler()),                # fitted on training data only
    ("pca", PCA(n_components=10, random_state=42)),
    ("clf", LogisticRegression(max_iter=5000, random_state=42)),
])
model.fit(X_train, y_train)                     # all three steps see X_train only
y_pred = model.predict(X_test)                  # test data is only transformed

pca = model.named_steps["pca"]
print(f"Train / test samples : {len(X_train)} / {len(X_test)}")
print(f"CEV(q = 10)          : {pca.explained_variance_ratio_.sum():.2%}")
print(f"Test accuracy        : {accuracy_score(y_test, y_pred):.2%}")
print(f"Malignant recall     : {recall_score(y_test, y_pred, pos_label=0):.2%}")
