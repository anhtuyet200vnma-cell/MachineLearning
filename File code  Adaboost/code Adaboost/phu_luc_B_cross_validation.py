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
    models = {'Decision stump': stump(), 'AdaBoost': ada()}
    # B: Cross-validation uses only the training partition.
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42)
        .split(Xtr,ytr,groups[train]))
    metrics = ['accuracy','precision','recall','f1']
    result['cv'] = {}
    for name in ('Decision stump','AdaBoost'):
        cv = cross_validate(models[name],Xtr,ytr,cv=folds,scoring=metrics)
        result['cv'][name] = {m:[float(cv['test_'+m].mean()),
            float(cv['test_'+m].std(ddof=1))] for m in metrics}
        print('CV',name,result['cv'][name],flush=True)
    result['grid'] = []
    for rate in (0.5,1.0):
        for n in (50,100,200):
            cv = cross_validate(ada(n,rate),Xtr,ytr,cv=folds,
                                scoring=['accuracy','f1'])
            row = {'rate':rate,'n':n,'accuracy':float(cv['test_accuracy'].mean()),
                'accuracy_sd':float(cv['test_accuracy'].std(ddof=1)),
                'f1':float(cv['test_f1'].mean())}
            result['grid'].append(row)
            print('GRID',row,flush=True)
    save_result('results_B.json', result)


if __name__ == "__main__":
    main()
