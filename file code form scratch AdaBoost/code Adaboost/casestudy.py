
import numpy as np

from common import AdaBoostFromScratch, classification_metrics, load_data, save_result, split


def main():
    X, y, groups, _, result, output_dir = load_data(__doc__)
    train, test = split(X, y, groups, seed=42)
    model = AdaBoostFromScratch(n_estimators=100, learning_rate=0.5).fit(X[train], y[train])
    prediction = model.predict(X[test])
    metrics = classification_metrics(y[test], prediction)

    for key in ('accuracy', 'precision', 'recall', 'f1'):
        print(f'{key.capitalize()}: {metrics[key]:.2%}')
    print('Confusion matrix [0, 1]:', metrics['confusion_matrix'])
    print('Accepted learners:', len(model.estimators_))
    print('First round:', model.history_[0])

    result.update({
        'data': {'features': X.shape[1], 'train': len(train), 'test': len(test),
                 'test_classes': np.bincount(y[test], minlength=2).tolist(), 'group_overlap': 0},
        'config': {'n_estimators': 100, 'learning_rate': 0.5, 'max_depth': 1, 'split_seed': 42},
        'metrics': metrics,
        'accepted_learners': len(model.estimators_),
        'stop_reason': model.stop_reason_,
        'first_round': model.history_[0],
    })
    save_result(output_dir, 'results_casestudy.json', result)


if __name__ == '__main__':
    main()
