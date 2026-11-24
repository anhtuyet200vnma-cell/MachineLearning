
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import numbers
from pathlib import Path
import platform
import zipfile

import numpy as np


def _features(X, n_features=None):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError('X must be a nonempty two-dimensional array.')
    if not np.isfinite(X).all():
        raise ValueError('X must contain only finite values.')
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError('The feature count differs from the fitted model.')
    return X


def _labels(y, n_samples):
    y = np.asarray(y)
    if y.shape != (n_samples,) or not np.isin(y, [0, 1]).all():
        raise ValueError('y must contain one binary label (0 or 1) per row.')
    return y.astype(np.int64)


def _weights(weights, n_samples):
    if weights is None:
        return np.full(n_samples, 1.0 / n_samples)
    weights = np.asarray(weights, dtype=np.float64).copy()
    if weights.shape != (n_samples,) or not np.isfinite(weights).all():
        raise ValueError('Weights must be a finite vector with one value per row.')
    if (weights < 0).any() or weights.max() <= 0:
        raise ValueError('Weights must be nonnegative with a positive total.')
    # Scaling before summing also accepts very large finite weights safely.
    weights /= weights.max()
    return weights / weights.sum()


def weighted_error(y, prediction, weights):
    """epsilon = sum_i w_i * I[y_i != prediction_i] / sum_i w_i."""
    y = np.asarray(y)
    if y.ndim != 1 or y.size == 0:
        raise ValueError('y must be a nonempty vector of binary labels.')
    y = _labels(y, len(y))
    prediction = _labels(prediction, len(y))
    weights = _weights(weights, len(y))
    return float(np.dot(weights, prediction != y))


def samme_coefficient(error, learning_rate=0.5):
    """For K=2: alpha = eta * ln((1-epsilon) / epsilon)."""
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be positive and finite.')
    if not np.isfinite(error) or not 0 < error < 0.5:
        raise ValueError('The ordinary SAMME step requires 0 < error < 0.5.')
    # The log difference remains finite even when (1-error)/error overflows.
    log_odds = np.log1p(-error) - np.log(error)
    with np.errstate(over='ignore', invalid='ignore'):
        alpha = float(learning_rate * log_odds)
    if not np.isfinite(alpha):
        raise FloatingPointError('The voting coefficient overflowed; reduce learning_rate.')
    return alpha


def update_weights(weights, incorrect, alpha):
    """w_new is proportional to w * exp(alpha * I[incorrect])."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = _weights(weights, len(weights))
    incorrect = np.asarray(incorrect, dtype=bool)
    if incorrect.shape != weights.shape or not np.isfinite(alpha):
        raise ValueError('Invalid error mask or voting coefficient.')
    positive = weights > 0
    log_weights = np.full(weights.shape, -np.inf)
    log_weights[positive] = np.log(weights[positive])
    log_weights[positive] += alpha * incorrect[positive]
    # A common multiplier cancels during normalization; avoid overflow.
    shifted = np.exp(log_weights - log_weights.max())
    total = shifted.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError('The sample weights cannot be normalized.')
    return shifted / total


@dataclass
class DecisionStumpFromScratch:
    """One split, chosen by minimizing weighted child-node Gini impurity.

    Leaf label: weighted majority, choosing 0 on an exact tie.
    Split tie: first feature and first threshold in ascending order.
    """
    feature_: int | None = None
    threshold_: float | None = None
    left_class_: int = 0
    right_class_: int = 0

    def fit(self, X, y, sample_weight=None):
        X = _features(X)
        y = _labels(y, len(X))
        weights = _weights(sample_weight, len(X))
        self._fit_arrays(X, y, weights, self._sort_features(X))
        return self

    @staticmethod
    def _sort_features(X):
        return [np.argsort(X[:, j], kind='stable') for j in range(X.shape[1])]

    def _fit_arrays(self, X, y, weights, orders):
        self.n_features_in_ = X.shape[1]
        self.feature_, self.threshold_ = None, None
        totals = np.bincount(y, weights=weights, minlength=2)
        majority = int(np.argmax(totals))
        self.left_class_ = self.right_class_ = majority
        total_weight = float(totals.sum())
        if totals.min() <= 0:
            return
        parent_gini = 1.0 - float(np.sum((totals / total_weight) ** 2))
        best_impurity = parent_gini
        for feature, order in enumerate(orders):
            # Zero-weight rows cannot create leaves supported by no data.
            order = order[weights[order] > 0]
            values, labels, w = X[order, feature], y[order], weights[order]
            positions = np.flatnonzero(values[:-1] < values[1:])
            if positions.size == 0:
                continue
            left0 = np.cumsum(w * (labels == 0))[positions]
            left1 = np.cumsum(w * (labels == 1))[positions]
            right0 = np.maximum(totals[0] - left0, 0.0)
            right1 = np.maximum(totals[1] - left1, 0.0)
            left_weight, right_weight = left0 + left1, right0 + right1
            # W_L*Gini(L) + W_R*Gini(R), divided by total weight.
            valid = (left_weight > 0) & (right_weight > 0)
            impurities = np.full(len(positions), np.inf)
            impurities[valid] = (
                2.0 * left0[valid] * left1[valid] / left_weight[valid]
                + 2.0 * right0[valid] * right1[valid] / right_weight[valid]
            ) / total_weight
            k = int(np.argmin(impurities))
            # Tolerance gives a deterministic rule for numerically tied splits.
            if impurities[k] < best_impurity - 1e-14:
                position = positions[k]
                lo, hi = values[position], values[position + 1]
                threshold = lo / 2.0 + hi / 2.0
                if threshold >= hi:  # Adjacent floating-point values.
                    threshold = lo
                best_impurity = float(impurities[k])
                self.feature_ = feature
                self.threshold_ = float(threshold)
                self.left_class_ = int(left1[k] > left0[k])
                self.right_class_ = int(right1[k] > right0[k])

    def predict(self, X):
        if not hasattr(self, 'n_features_in_'):
            raise ValueError('Fit the stump before predicting.')
        X = _features(X, self.n_features_in_)
        if self.feature_ is None:
            return np.full(len(X), self.left_class_, dtype=np.int64)
        return np.where(X[:, self.feature_] <= self.threshold_,
                        self.left_class_, self.right_class_)


class AdaBoostFromScratch:
    """Binary SAMME: own stump training, errors, updates and weighted votes."""

    def __init__(self, n_estimators=100, learning_rate=0.5):
        if (not isinstance(n_estimators, numbers.Integral)
                or isinstance(n_estimators, bool) or n_estimators < 1):
            raise ValueError('n_estimators must be a positive integer.')
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite.')
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)

    def fit(self, X, y, sample_weight=None):
        X = _features(X)
        y = _labels(y, len(X))
        if np.unique(y).size != 2:
            raise ValueError('Binary SAMME requires both labels 0 and 1.')
        weights = _weights(sample_weight, len(X))  # w_i^(1) = 1/N by default.
        zero_weights = weights == 0
        orders = DecisionStumpFromScratch._sort_features(X)
        self.n_features_in_ = X.shape[1]
        self.estimators_, self.estimator_errors_, self.estimator_weights_ = [], [], []
        self.history_ = []
        self.stop_reason_ = 'maximum_rounds'
        self.first_weights_before_ = weights.copy()
        alpha_sum = 0.0
        for t in range(self.n_estimators):
            # Positive weights cannot underflow to zero before fitting a stump.
            weights = np.maximum(weights, np.finfo(np.float64).eps)
            weights[zero_weights] = 0.0
            weights /= weights.sum()
            stump = DecisionStumpFromScratch()
            stump._fit_arrays(X, y, weights, orders)
            prediction = stump.predict(X)
            incorrect = prediction != y
            error = weighted_error(y, prediction, weights)
            # Binary SAMME requires epsilon < 1 - 1/K = 1/2.
            # Roundoff can turn an exact 0.5 into 0.4999999999999999.
            if error >= 0.5 - 8 * np.finfo(np.float64).eps:
                self.stop_reason_ = 'weak_learner_rejected'
                if not self.estimators_:
                    raise ValueError('No weak learner has weighted error below 0.5.')
                break
            # Explicit finite convention for epsilon=0, also used in sklearn 1.8.
            alpha = 1.0 if error <= 0 else samme_coefficient(error, self.learning_rate)
            if not np.isfinite(alpha_sum + alpha):
                raise FloatingPointError('The total voting weight overflowed; reduce learning_rate.')
            alpha_sum += alpha
            self.estimators_.append(stump)
            self.estimator_errors_.append(error)
            self.estimator_weights_.append(alpha)
            if t == 0:
                self.first_weights_before_ = weights.copy()
                self.first_prediction_ = prediction.copy()
                self.first_weights_after_ = (
                    weights.copy() if error <= 0 else update_weights(weights, incorrect, alpha)
                )
            self.history_.append({'round': t + 1, 'error': error, 'alpha': alpha,
                                  'feature': stump.feature_, 'threshold': stump.threshold_})
            if error <= 0:
                self.stop_reason_ = 'perfect_learner'
                break
            if t + 1 < self.n_estimators:
                weights = update_weights(weights, incorrect, alpha)
        self.estimator_errors_ = np.asarray(self.estimator_errors_)
        self.estimator_weights_ = np.asarray(self.estimator_weights_)
        return self

    def decision_function(self, X):
        if not getattr(self, 'estimators_', None):
            raise ValueError('Fit AdaBoost successfully before predicting.')
        X = _features(X, self.n_features_in_)
        score = np.zeros(len(X))
        for alpha, stump in zip(self.estimator_weights_, self.estimators_):
            # Class-1 vote minus class-0 vote: alpha * (2*h(x) - 1).
            score += alpha * (2 * stump.predict(X) - 1)
        return score

    def predict(self, X):
        # A zero score chooses class 0, consistently with the leaf tie rule.
        return (self.decision_function(X) > 0).astype(np.int64)


def classification_metrics(y_true, y_pred):
    """Compute binary confusion matrix, Accuracy, Precision, Recall, F1 in NumPy."""
    truth = np.asarray(y_true)
    if truth.ndim != 1 or truth.size == 0:
        raise ValueError('y_true must be a nonempty vector.')
    truth = _labels(truth, len(truth))
    pred = _labels(y_pred, len(truth))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    tp = int(np.sum((truth == 1) & (pred == 1)))
    return {'accuracy': (tn + tp) / len(truth),
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / (tp + fn) if tp + fn else 0.0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            'confusion_matrix': [[tn, fp], [fn, tp]]}


def group_folds(X, y, groups, seed=42):
    """Tạo 5 fold theo nhóm, cùng thiết lập với thí nghiệm gốc."""
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, test in splitter.split(X, y, groups):
        if np.intersect1d(groups[train], groups[test]).size:
            raise ValueError('Có nhóm đặc trưng xuất hiện trong cả hai tập.')
        yield train, test


def split(X, y, groups, seed=42):
    """Lấy fold đầu tiên làm tập test giữ riêng."""
    return next(group_folds(X, y, groups, seed))


def row_split(y, seed=42):
    """Tạo phép chia theo dòng cho phép đối chứng ở Phụ lục D."""
    from sklearn.model_selection import train_test_split

    return train_test_split(np.arange(len(y)), test_size=0.2, stratify=y,
                            random_state=seed)


def load_data(description):
    """Đọc dữ liệu và lưu thông tin cần để tái lập thí nghiệm."""
    import pandas as pd
    import sklearn

    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data', type=Path, help='Đường dẫn spambase.data hoặc spambase.zip')
    parser.add_argument('--output-dir', type=Path, default=base / 'outputs_from_scratch',
                        help='Thư mục ghi kết quả')
    args = parser.parse_args()
    path = args.data or base / 'spambase.data'
    if args.data is None and not path.is_file():
        path = base / 'spambase.zip'
    if not path.is_file():
        parser.error('Đặt spambase.data hoặc spambase.zip cạnh code, hoặc chỉ đường dẫn qua --data.')

    try:
        if path.suffix.lower() == '.zip':
            with zipfile.ZipFile(path) as archive:
                entries = [entry for entry in archive.infolist()
                           if not entry.is_dir() and Path(entry.filename).name == 'spambase.data']
                if len(entries) != 1:
                    parser.error('File ZIP cần chứa đúng một file spambase.data.')
                data_bytes = archive.read(entries[0])
        else:
            data_bytes = path.read_bytes()
        raw = pd.read_csv(io.BytesIO(data_bytes), header=None)
        if raw.shape[1] != 58 or raw.isna().to_numpy().any():
            raise ValueError('Cần 57 cột đặc trưng và 1 cột nhãn, không có giá trị thiếu.')
        features = raw.iloc[:, :-1].astype(np.float64)
        X = _features(features)
        y = _labels(raw.iloc[:, -1].to_numpy(), len(X))
        if np.unique(y).size != 2:
            raise ValueError('Dữ liệu cần có cả nhãn 0 và nhãn 1.')
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(f'Không đọc được dữ liệu ở {path}. {exc}')

    # Giữ cách tạo nhóm giống bản thư viện để so sánh trên cùng các fold.
    groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
    row_hashes = pd.util.hash_pandas_object(raw, index=False).to_numpy()
    result = {
        'implementation': 'Python + NumPy: weighted Gini stump and binary SAMME',
        'environment': {'Python': platform.python_version(), 'scikit-learn': sklearn.__version__,
                        'pandas': pd.__version__, 'NumPy': np.__version__},
        'run_date_utc': datetime.now(timezone.utc).isoformat(),
        'sha256': hashlib.sha256(data_bytes).hexdigest(),
        'rows': len(y),
        'data_path': str(path.resolve()),
    }
    return X, y, groups, row_hashes, result, args.output_dir


def save_result(output_dir, filename, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding='utf-8')
    print(f'Đã lưu {path}')
