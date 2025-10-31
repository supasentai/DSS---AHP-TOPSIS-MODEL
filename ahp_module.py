import pandas as pd
import numpy as np
import yaml
import os

def calculate_ahp_weights(matrix):
    matrix = np.array(matrix, dtype=float)
    n = matrix.shape[0]
    if n != matrix.shape[1]: return None, None
    col_sums = matrix.sum(axis=0)
    normalized_matrix = matrix / col_sums
    weights = normalized_matrix.mean(axis=1)
    ri_values = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    if n not in ri_values: return weights, None
    weighted_sum_vector = np.dot(matrix, weights)
    lambda_max = np.mean(weighted_sum_vector / weights)
    ci = (lambda_max - n) / (n - 1)
    ri = ri_values[n]
    cr = ci / ri if ri != 0 else 0
    return weights, cr


def save_weights_to_yaml(weights, criteria_names, model_name, filename="weights.yaml"):
    new_model_weights = {name: round(float(weight), 4) for name, weight in zip(criteria_names, weights)}
    all_models_data = {}
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            all_models_data = yaml.safe_load(f) or {}
    all_models_data[model_name] = new_model_weights
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(all_models_data, f, allow_unicode=True, sort_keys=False, indent=2)
    print("\n" + "=" * 50)
    print(f"--- CẬP NHẬT FILE THÀNH CÔNG ---")
    print(f"Đã thêm/cập nhật mô hình '{model_name}' vào file '{filename}'.")
    print("=" * 50)


def build_comparison_matrix_interactively(criteria_names):
    n = len(criteria_names)
    matrix = np.ones((n, n))
    print("\n" + "=" * 70)
    print("--- BẮT ĐẦU QUÁ TRÌNH SO SÁNH CẶP TƯƠNG TÁC ---")
    print("=" * 70)
    for i in range(n):
        for j in range(i + 1, n):
            while True:
                try:
                    prompt = f"-> So sánh '{criteria_names[i]}' với '{criteria_names[j]}': "
                    value_str = input(prompt).strip()
                    if '/' in value_str:
                        num, den = value_str.split('/')
                        value = float(num) / float(den)
                    else:
                        value = float(value_str)
                    if value <= 0:
                        print("Lỗi: Vui lòng nhập một số dương.")
                        continue
                    matrix[i, j] = value
                    matrix[j, i] = 1 / value
                    break
                except (ValueError, ZeroDivisionError):
                    print("Lỗi: Input không hợp lệ.")
    return matrix


def run_interactive_ahp():
    """Hàm chính để chạy toàn bộ quy trình AHP tương tác."""
    csv_data_filename = "AHP_Data_synced_fixed.csv"
    output_yaml_filename = "weights.yaml"

    try:
        df_data = pd.read_csv(csv_data_filename, encoding='latin1')
        criteria_names = [col for col in df_data.columns if col not in ['ward', 'ward_id']]
        print(f"Đã tìm thấy {len(criteria_names)} tiêu chí để phân tích.")

        model_name = input("\nVui lòng nhập tên cho mô hình cần tạo/cập nhật: ").strip().lower()
        if not model_name:
            print("Lỗi: Tên mô hình không được để trống.")
            return

        while True:
            comparison_matrix = build_comparison_matrix_interactively(criteria_names)
            print("\nMa trận so sánh cặp đã hoàn thành:")
            df_display = pd.DataFrame(comparison_matrix, index=criteria_names, columns=criteria_names)
            print(df_display.round(3))

            priority_weights, consistency_ratio = calculate_ahp_weights(comparison_matrix)

            if priority_weights is not None and consistency_ratio is not None and consistency_ratio < 0.10:
                print(f"\nKiểm tra nhất quán: TỐT (CR = {consistency_ratio:.4f})")
                save_weights_to_yaml(priority_weights, criteria_names, model_name, output_yaml_filename)
                break
            else:
                cr_value = f"{consistency_ratio:.4f}" if consistency_ratio is not None else "N/A"
                print(f"\nCẢNH BÁO: Tỷ số nhất quán (CR) là {cr_value} (>= 0.1).")
                retry = input("Bạn có muốn nhập lại các giá trị không? (y/n): ").strip().lower()
                if retry != 'y':
                    print("Đã hủy quá trình.")
                    break
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu '{csv_data_filename}'.")
    except Exception as e:
        print(f"Đã xảy ra một lỗi không mong muốn: {e}")


if __name__ == '__main__':
    # Cho phép chạy file này độc lập để test
    run_interactive_ahp()