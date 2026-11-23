import json
from pathlib import Path
import numpy as np

def du_doan_mau_moi(X, mo_hinh):
    X = np.asarray(X, dtype=np.float32)
    if (
        X.ndim != 2
        or X.shape[1] != mo_hinh["n_features"]
        or not np.isfinite(X).all()
    ):
        raise ValueError("Dữ liệu mới không hợp lệ.")
    ket_qua = []
    for mau in X:
        node = mo_hinh["root"]

        while node["left"] is not None:
            if mau[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]

        ket_qua.append(node["label"])
    return np.asarray(ket_qua)
if __name__ == "__main__":
   
    duong_dan = Path(__file__).resolve().parent / "cay_wine.json"

    if not duong_dan.is_file():
        raise FileNotFoundError(
            "Chưa có cay_wine.json. Hãy chạy code huấn luyện trước."
        )

    with duong_dan.open(encoding="utf-8") as f:
        mo_hinh = json.load(f)

    mau_moi = [
        [3.00],
        [5.00],
        [9.00]
    ]

    ket_qua = du_doan_mau_moi(mau_moi, mo_hinh)

    for mau, nhom in zip(mau_moi, ket_qua):
        print(
            f"Cường độ màu {mau[0]:.2f}"
            f" -> Dự đoán nhóm {int(nhom)}"
        )