import numpy as np

from common import (
    AdaBoostFromScratch, DecisionStumpFromScratch, classification_metrics,
    group_folds, load_data, save_result, split,
)


METRICS = ('accuracy', 'precision', 'recall', 'f1')


def score_folds(make_model, X, y, folds):
    scores = {name: [] for name in METRICS}
    for train, validation in folds:
        model = make_model().fit(X[train], y[train])
        metrics = classification_metrics(y[validation], model.predict(X[validation]))
        for name in METRICS:
            scores[name].append(metrics[name])
    return {name: [float(np.mean(values)), float(np.std(values, ddof=1))]
            for name, values in scores.items()}


def main():
    X, y, groups, _, result, output_dir = load_data(__doc__)
    train, _ = split(X, y, groups, seed=42)
    # CV và tìm tham số chỉ dùng tập train, không dùng tập test cuối.
    X_train, y_train = X[train], y[train]
    folds = list(group_folds(X_train, y_train, groups[train], seed=42))
    stump_cv = score_folds(DecisionStumpFromScratch, X_train, y_train, folds)
    ada_cv = score_folds(AdaBoostFromScratch, X_train, y_train, folds)
    result['cv'] = {'Decision stump': stump_cv, 'AdaBoost': ada_cv}
    for name, scores in result['cv'].items():
        print('CV', name, scores, flush=True)

    result['grid'] = []
    for rate in (0.5, 1.0):
        for n in (50, 100, 200):
            if (rate, n) == (0.5, 100):
                scores = ada_cv
            else:
                scores = score_folds(lambda: AdaBoostFromScratch(n, rate), X_train, y_train, folds)
            accuracy, accuracy_sd = scores['accuracy']
            row = {'rate': rate, 'n': n, 'accuracy': accuracy,
                   'accuracy_sd': accuracy_sd, 'f1': scores['f1'][0]}
            result['grid'].append(row)
            print('GRID', row, flush=True)
    save_result(output_dir, 'results_B.json', result)


if __name__ == '__main__':
    main()