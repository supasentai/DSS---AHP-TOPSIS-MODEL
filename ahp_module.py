import os
import yaml
import numpy as np


def calculate_ahp_weights(matrix):
    """
    Tính vector trọng số (AHP) và hệ số nhất quán CR từ ma trận so sánh cặp.

    Parameters
    ----------
    matrix : array-like (n x n)
        Ma trận so sánh cặp AHP.

    Returns
    -------
    weights : np.ndarray | None
        Vector trọng số chuẩn hóa theo cột, lấy trung bình theo hàng.
    cr : float | None
        Consistency Ratio. Trả về None nếu không có RI cho kích thước n.
    """
    matrix = np.array(matrix, dtype=float)
    n = matrix.shape[0]

    # Kiểm tra ma trận vuông
    if n != matrix.shape[1]:
        print("Lỗi: Ma trận phải là ma trận vuông.")
        return None, None

    # Chuẩn hóa theo cột
    col_sums = matrix.sum(axis=0)
    if np.any(col_sums == 0):
        print("Lỗi: Có cột có tổng bằng 0, không thể chuẩn hóa.")
        return None, None

    normalized = matrix / col_sums
    weights = normalized.mean(axis=1)

    # Bảng RI (Saaty)
    ri_values = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
        11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59
    }

    # Không có RI cho kích thước này → chỉ trả trọng số
    if n not in ri_values:
        print(f"Cảnh báo: Không có chỉ số RI cho ma trận kích thước {n}. Bỏ qua kiểm tra nhất quán.")
        return weights, None

    # Trọng số có phần tử 0 → không tính được lambda_max an toàn
    if np.any(weights == 0):
        print("Lỗi: Vector trọng số chứa giá trị 0, không thể tính Lambda-max.")
        return weights, 0.0

    weighted_sum = np.dot(matrix, weights)
    lambda_max = np.mean(weighted_sum / weights)

    ci = (lambda_max - n) / (n - 1) if (n - 1) != 0 else 0.0
    ri = ri_values[n]
    cr = ci / ri if ri != 0 else 0.0

    return weights, cr


def save_weights_to_yaml(weights_dict, model_name, filename="data/weights.yaml"):
    """
    Lưu một mô hình trọng số vào YAML. Nếu tên đã tồn tại, tự động thêm _1, _2, ...

    Parameters
    ----------
    weights_dict : dict
        {criterion_name: weight_value}
    model_name : str
        Tên mô hình cần lưu.
    filename : str
        Đường dẫn file YAML.

    Returns
    -------
    success : bool
        True nếu ghi file thành công.
    saved_name : str | None
        Tên mô hình đã được dùng để lưu (có thể khác với model_name nếu trùng).
    """
    all_models = {}
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                all_models = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Cảnh báo: Không đọc được '{filename}'. Sẽ tạo file mới. Lỗi: {e}")
            all_models = {}

    # Tạo tên duy nhất nếu bị trùng
    final_name = model_name.strip()
    if final_name in all_models:
        i = 1
        while f"{final_name}_{i}" in all_models:
            i += 1
        print(f"Thông báo: Tên '{final_name}' đã tồn tại. Lưu với tên mới: '{final_name}_{i}'")
        final_name = f"{final_name}_{i}"

    # Ép kiểu an toàn trước khi ghi
    cleaned = {str(k): float(v) for k, v in weights_dict.items()}
    all_models[final_name] = cleaned

    try:
        with open(filename, "w", encoding="utf-8") as f:
            yaml.dump(all_models, f, allow_unicode=True, sort_keys=False, indent=2)
        return True, final_name
    except Exception as e:
        print(f"Lỗi: Không thể ghi file '{filename}'. {e}")
        return False, None
