from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

                                
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=2,
    ccp_alpha=0.02,
    random_state=42
)
model.fit(X_train, y_train)


mau_moi = [
    [3.00],
    [5.00],
    [9.00]
]

du_doan = model.predict(mau_moi)

for mau, nhom in zip(mau_moi, du_doan):
    print(
        f"Cường độ màu {mau[0]:.2f}"
        f" -> Dự đoán nhóm {int(nhom)}"
    )