import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# 1. Load the dataset: 57 features and one class label.
data = pd.read_csv(r"c:\code\spambase\spambase.data", header=None)
X = data.iloc[:, :-1].astype("float64")
y = data.iloc[:, -1].astype("int64").to_numpy()

# 2. Keep identical feature vectors in the same group.
groups = pd.util.hash_pandas_object(X, index=False).to_numpy()

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 3. Configure AdaBoost with decision stumps.
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42,
    ),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42,
)

# 4. Train the model and predict the test labels.
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Evaluate performance, with spam (label 1) as positive.
scores = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1": f1_score(y_test, y_pred, zero_division=0),
}

for metric, value in scores.items():
    print(f"{metric}: {value:.2%}")

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("Accepted learners:", len(model.estimators_))