"""Tách từ phụ lục của Adaboost.docx; chạy độc lập với file case study."""
from common import load_data, split, stump, ada, save_result
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    raw, X, y, groups, result = load_data()
    train, test = split(X, y, groups)
    assert not np.intersect1d(groups[train], groups[test]).size
    Xtr, Xte, ytr, yte = X.iloc[train], X.iloc[test], y[train], y[test]
    preds = {name: model.fit(Xtr, ytr).predict(Xte)
             for name, model in {'Decision stump': stump(), 'AdaBoost': ada()}.items()}
    # C: Paired bootstrap over test groups, with fixed fitted predictions.
    unique, inverse = np.unique(groups[test],return_inverse=True)
    counts = np.bincount(inverse)
    difference = (preds['AdaBoost']==yte).astype(int) - (preds['Decision stump']==yte)
    sums = np.bincount(inverse,weights=difference)
    rng = np.random.default_rng(42)
    boot = np.empty(4000)
    for b in range(len(boot)):
        selected = rng.integers(0,len(unique),size=len(unique))
        boot[b] = sums[selected].sum()/counts[selected].sum()
    result['bootstrap'] = {'seed':42,'resamples':4000,
        'ci':np.quantile(boot,[0.025,0.975],method='linear').tolist()}
    print('95% CI (percentage points):', [100*v for v in result['bootstrap']['ci']])
    save_result('results_C.json', result)


if __name__ == "__main__":
    main()
