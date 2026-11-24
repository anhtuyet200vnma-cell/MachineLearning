import numpy as np

from common import AdaBoostFromScratch, DecisionStumpFromScratch, load_data, save_result, split


def main():
    X, y, groups, _, result, output_dir = load_data(__doc__)
    train, test = split(X, y, groups, seed=42)
    stump = DecisionStumpFromScratch().fit(X[train], y[train])
    boost = AdaBoostFromScratch().fit(X[train], y[train])
    truth = y[test]
    stump_correct = stump.predict(X[test]) == truth
    boost_correct = boost.predict(X[test]) == truth
    difference = boost_correct.astype(int) - stump_correct.astype(int)

    unique_groups, group_indices = np.unique(groups[test], return_inverse=True)
    counts = np.bincount(group_indices)
    sums = np.bincount(group_indices, weights=difference)
    seed, resamples = 42, 4000
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples)
    for i in range(resamples):
        selected = rng.integers(0, len(unique_groups), size=len(unique_groups))
        bootstrap[i] = sums[selected].sum() / counts[selected].sum()
    interval = np.quantile(bootstrap, [0.025, 0.975], method='linear').tolist()

    result['bootstrap'] = {'seed': seed, 'resamples': resamples, 'ci': interval}
    print(f'95% CI for the accuracy difference: {100 * interval[0]:.2f} '
          f'to {100 * interval[1]:.2f} percentage points')
    save_result(output_dir, 'results_C.json', result)


if __name__ == '__main__':
    main()