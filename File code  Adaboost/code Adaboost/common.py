"""Hàm dùng chung cho Phụ lục B, C, D của Adaboost.docx."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
import platform
import zipfile

import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold


def stump(seed=42):
    return DecisionTreeClassifier(max_depth=1, random_state=seed)


def ada(n=100, rate=0.5, seed=42):
    return AdaBoostClassifier(estimator=stump(seed), n_estimators=n,
                              learning_rate=rate, random_state=seed)


def split(X, y, groups, seed=42):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    return next(cv.split(X, y, groups))


def load_data():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='Đường dẫn spambase.data hoặc spambase.zip')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    path = args.data or base / 'spambase.data'
    if args.data is None and not path.exists():
        path = base / 'spambase.zip'
    if not path.is_file():
        parser.error('Thiếu dữ liệu. Đặt spambase.data hoặc spambase.zip cạnh các file code, hoặc dùng --data.')
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if Path(n).name == 'spambase.data']
            if len(names) != 1:
                parser.error('ZIP cần chứa đúng một file spambase.data.')
            data_bytes = archive.read(names[0])
    else:
        data_bytes = path.read_bytes()
    raw = pd.read_csv(io.BytesIO(data_bytes), header=None)
    if raw.shape[1] != 58 or raw.isna().any().any():
        parser.error('Dữ liệu phải có 57 đặc trưng và 1 cột nhãn, không có giá trị thiếu.')
    if set(raw.iloc[:, -1].unique()) != {0, 1}:
        parser.error('Nhãn dữ liệu phải gồm 0 và 1.')
    X = raw.iloc[:, :-1].astype('float64')
    y = raw.iloc[:, -1].astype('int64').to_numpy()
    if not np.isfinite(X.to_numpy()).all():
        parser.error('Đặc trưng chứa giá trị không hữu hạn.')
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    metadata = {
        'environment': {'Python': platform.python_version(),
                        'scikit-learn': sklearn.__version__,
                        'pandas': pd.__version__, 'NumPy': np.__version__},
        'run_date_utc': datetime.now(timezone.utc).isoformat(),
        'sha256': hashlib.sha256(data_bytes).hexdigest(),
        'rows': len(raw), 'data_path': str(path.resolve()),
    }
    return raw, X, y, groups, metadata


def save_result(filename, result):
    folder = Path(__file__).resolve().parent / 'outputs'
    folder.mkdir(exist_ok=True)
    path = folder / filename
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Đã lưu: {path}')
