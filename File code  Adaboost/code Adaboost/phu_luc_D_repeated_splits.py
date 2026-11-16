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
    # D: Repeat splitting for seeds 0 through 11; hold the learner seed at 42.
    result['seeds'] = []
    xy = pd.util.hash_pandas_object(raw,index=False).to_numpy()
    for seed in range(12):
        rt,re = train_test_split(np.arange(len(y)),test_size=0.2,
                                stratify=y,random_state=seed)
        gt,ge = split(X, y, groups, seed)
        a = ada().fit(X.iloc[rt],y[rt])
        b = ada().fit(X.iloc[gt],y[gt])
        row = {'seed':seed,
            'row_accuracy':float(accuracy_score(y[re],a.predict(X.iloc[re]))),
            'group_accuracy':float(accuracy_score(y[ge],b.predict(X.iloc[ge]))),
            'row_overlap_xy':int(np.isin(xy[re],xy[rt]).sum()),
            'group_test_size':len(ge)}
        result['seeds'].append(row)
        print('SEED',row,flush=True)
    rt,re = train_test_split(np.arange(len(y)),test_size=0.2,
                            stratify=y,random_state=42)
    a = ada().fit(X.iloc[rt],y[rt])
    result['row_seed42'] = {'overlap_xy':int(np.isin(xy[re],xy[rt]).sum()),
        'cm':confusion_matrix(y[re],a.predict(X.iloc[re]),labels=[0,1]).tolist()}
    result['summary'] = {k: {'mean': float(np.mean([r[k] for r in result['seeds']])), 'sd': float(np.std([r[k] for r in result['seeds']], ddof=1))} for k in ('row_accuracy', 'group_accuracy')}
    print(result['summary'])
    save_result('results_D.json', result)


if __name__ == "__main__":
    main()
