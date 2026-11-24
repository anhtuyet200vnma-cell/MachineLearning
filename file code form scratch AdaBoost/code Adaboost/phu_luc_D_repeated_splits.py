import numpy as np

from common import AdaBoostFromScratch, classification_metrics, load_data, row_split, save_result, split


def predict_on_split(X, y, train, test):
    model = AdaBoostFromScratch().fit(X[train], y[train])
    return model.predict(X[test])


def main():
    X, y, groups, row_hashes, result, output_dir = load_data(__doc__)
    result['seeds'] = []
    for seed in range(12):
        row_train, row_test = row_split(y, seed)
        group_train, group_test = split(X, y, groups, seed)
        row_prediction = predict_on_split(X, y, row_train, row_test)
        group_prediction = predict_on_split(X, y, group_train, group_test)
        row = {
            'seed': seed,
            'row_accuracy': float(np.mean(row_prediction == y[row_test])),
            'group_accuracy': float(np.mean(group_prediction == y[group_test])),
            'row_overlap_xy': int(np.isin(row_hashes[row_test], row_hashes[row_train]).sum()),
            'group_test_size': len(group_test),
        }
        result['seeds'].append(row)
        print('SEED', row, flush=True)

    train, test = row_split(y, seed=42)
    prediction = predict_on_split(X, y, train, test)
    result['row_seed42'] = {
        'overlap_xy': int(np.isin(row_hashes[test], row_hashes[train]).sum()),
        'cm': classification_metrics(y[test], prediction)['confusion_matrix'],
    }
    result['summary'] = {}
    for name in ('row_accuracy', 'group_accuracy'):
        accuracies = [row[name] for row in result['seeds']]
        result['summary'][name] = {'mean': float(np.mean(accuracies)),
                                   'sd': float(np.std(accuracies, ddof=1))}
    print(result['summary'])
    save_result(output_dir, 'results_D.json', result)


if __name__ == '__main__':
    main()
