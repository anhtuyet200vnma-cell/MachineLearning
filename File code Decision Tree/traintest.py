from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

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
  
y_pred = model.predict(X_test)
   
so_dung = (y_pred == y_test).sum()
so_sai = (y_pred != y_test).sum()
accuracy = accuracy_score(y_test, y_pred)
   
print("Số mẫu train:", len(y_train))
print("Số mẫu test:", len(y_test))
print("Dự đoán đúng:", so_dung)
print("Dự đoán sai:", so_sai)
print(f"Accuracy: {accuracy:.2%}")
      
print(export_text(
    model,
    feature_names=["Cuong_do_mau"],
    decimals=2
))               
from sklearn.metrics import confusion_matrix, classification_report

print("\nMa trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred, labels=[1, 2, 3]))

print("\nCác chỉ số đánh giá:")
print(classification_report(
    y_test,
    y_pred,
    labels=[1, 2, 3],
    target_names=["Nhóm 1", "Nhóm 2", "Nhóm 3"],
    digits=4,
    zero_division=0
))