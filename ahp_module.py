import numpy as np
import yaml
import os


def calculate_ahp_weights(matrix):
    """
    Tính toán vector trọng số và kiểm tra tính nhất quán từ ma trận so sánh cặp AHP.
    """
    matrix = np.array(matrix, dtype=float)
    n = matrix.shape[0]

    if n != matrix.shape[1]:
        print("Lỗi: Ma trận phải là ma trận vuông.")
        return None, None

    col_sums = matrix.sum(axis=0)
    # Thêm kiểm tra nếu tổng cột bằng 0
    if np.any(col_sums == 0):
        print("Lỗi: Tổng cột bằng 0, không thể chuẩn hóa.")
        return None, None

    normalized_matrix = matrix / col_sums
    weights = normalized_matrix.mean(axis=1)

    ri_values = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
        11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59
    }
    if n not in ri_values:
        print(f"Cảnh báo: Không có chỉ số RI cho ma trận kích thước {n}. Bỏ qua kiểm tra nhất quán.")
        return weights, None

    # Thêm kiểm tra nếu trọng số bằng 0
    if np.any(weights == 0):
        print("Lỗi: Vector trọng số chứa giá trị 0, không thể tính Lambda-max.")
        return weights, 0.0  # Trả về trọng số nhưng CR = 0

    weighted_sum_vector = np.dot(matrix, weights)
    # Thêm kiểm tra chia cho 0
    if np.any(weights == 0):
        print("Lỗi: Trọng số bằng 0, không thể tính lambda_max.")
        return weights, 0.0

    lambda_max = np.mean(weighted_sum_vector / weights)
    ci = (lambda_max - n) / (n - 1) if (n - 1) != 0 else 0
    ri = ri_values[n]
    cr = ci / ri if ri != 0 else 0

    return weights, cr


def save_weights_to_yaml(weights_dict, model_name, filename="weights.yaml"):
    """
    Cập nhật (thêm hoặc ghi đè) một mô hình trọng số vào file YAML
    mà không làm mất các mô hình khác.
    """
    all_models_data = {}
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                all_models_data = yaml.safe_load(f)
                if all_models_data is None:
                    all_models_data = {}
        except Exception as e:
            print(f"CẢNH BÁO: Không đọc được file '{filename}' cũ. Sẽ tạo file mới. Lỗi: {e}")
            all_models_data = {}

    # Cập nhật dictionary với mô hình mới
    cleaned_weights_dict = {str(k): float(v) for k, v in weights_dict.items()}
    all_models_data[model_name] = cleaned_weights_dict

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(all_models_data, f, allow_unicode=True, sort_keys=False, indent=2)
        return True  # Trả về True nếu thành công
    except Exception as e:
        print(f"LỖI: Không thể ghi ra file '{filename}'. Lỗi: {e}")
        return False  # Trả về False nếu thất bại

