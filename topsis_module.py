import pandas as pd
import numpy as np
import json
import os
import yaml

# ====================================================================
# --- CẤU HÌNH ĐƯỜNG DẪN VÀ TRỌNG SỐ AHP ---
# ====================================================================
CSV_PATH = os.path.join("AHP_Data_synced_fixed.csv")
JSON_PATH = os.path.join("metadata.json")
CRITERIA_WEIGHTS_PATH = os.path.join("weights.yaml")  # Đảm bảo tên file YAML này khớp với tên file bạn dùng



def run_topsis_model(csv_path, json_path, analysis_type, all_criteria_weights):
    """Hàm chính để chạy toàn bộ quy trình TOPSIS."""
    # ... (Toàn bộ nội dung của hàm run_topsis_model của bạn giữ nguyên ở đây) ...
    print(f"\n--- KHỞI CHẠY MÔ HÌNH TOPSIS CHO: {analysis_type.upper()} ---")

    criteria_weights = all_criteria_weights.get(analysis_type)
    if not criteria_weights:
        print(f"LỖI: Không tìm thấy trọng số cho '{analysis_type}'.")
        return

    # ... (các bước còn lại của TOPSIS) ...
# ====================================================================
# --- HÀM TẢI DỮ LIỆU ---
# ====================================================================

def load_metadata(json_path):
    """Tải metadata và trích xuất loại tiêu chí (benefit/cost)."""
    try:
        # Sử dụng 'utf-8-sig' để đảm bảo xử lý BOM
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            metadata = json.load(f)

        criteria_types = {k: v.get('type') for k, v in metadata.items() if v.get('type') in ['benefit', 'cost']}
        return criteria_types
    except Exception as e:
        print(f"Lỗi khi tải metadata: {e}")
        return None


def load_criteria_weights(yaml_path):
    """
    Tải trọng số tiêu chí (AHP Weights) từ file YAML.
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            # Sử dụng safe_load nếu file không chứa tag numpy
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file trọng số tiêu chí tại '{yaml_path}'.")
        return None
    except Exception as e:
        print(f"LỖI khi đọc file trọng số YAML: {e}")
        return None


def get_user_selection(all_weights):
    """
    Hiển thị các loại hình có sẵn và yêu cầu người dùng chọn qua input().
    """
    if not all_weights:
        print("Lỗi: Không có dữ liệu trọng số để chọn.")
        return None

    model_keys = list(all_weights.keys())

    print("\n" + "=" * 50)
    print("--- CHỌN MÔ HÌNH PHÂN TÍCH TOPSIS ---")
    print("=" * 50)
    for i, key in enumerate(model_keys):
        print(f"[{i + 1}] {key.upper()} ({key})")

    while True:
        try:
            choice = input("\nVui lòng nhập tên loại hình (ví dụ: office) hoặc số thứ tự: ").strip().lower()

            if choice in model_keys:
                return choice

            choice_index = int(choice) - 1
            if 0 <= choice_index < len(model_keys):
                return model_keys[choice_index]

            print("Lựa chọn không hợp lệ. Vui lòng nhập lại.")

        except ValueError:
            print("Lựa chọn không hợp lệ. Vui lòng nhập tên hoặc số thứ tự.")


# ====================================================================
# --- QUY TRÌNH CHÍNH TOPSIS (MAIN EXECUTION) ---
# ====================================================================

def run_topsis_model(csv_path, json_path, analysis_type, all_criteria_weights):
    print(f"\n--- KHỞI CHẠY MÔ HÌNH TOPSIS CHO: {analysis_type.upper()} ---")

    # --- BƯỚC 0: TẢI DỮ LIỆU VÀ TRỌNG SỐ ---

    # 1. Lấy trọng số AHP cho mô hình đã chọn
    criteria_weights = all_criteria_weights.get(analysis_type)
    if not criteria_weights:
        print(f"LỖI: Không tìm thấy trọng số tiêu chí cho '{analysis_type}'.")
        return

    # 2. Lấy phân loại Benefit/Cost
    criteria_types = load_metadata(json_path)
    if criteria_types is None: return

    # 3. Tải dữ liệu thô (CSV)
    try:
        # SỬA: Thêm encoding và decimal để xử lý dữ liệu tiếng Việt và dấu phẩy thập phân
        df = pd.read_csv(csv_path, encoding='latin1', decimal=',')

        # Chỉ giữ lại các tiêu chí có trong metadata và trọng số
        valid_criteria = [c for c in criteria_types.keys() if c in criteria_weights]

        df_locations = df[['ward', 'ward_id']].copy()  # Đã sửa 'ward' thành 'Ward' nếu cần
        df_data = df[valid_criteria].astype(float)

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu tại '{csv_path}'.")
        return
    except KeyError as e:
        print(f"Lỗi: Thiếu cột quan trọng trong CSV hoặc Metadata: {e}")
        return
    except Exception as e:
        print(f"Lỗi khi đọc file CSV hoặc xử lý dữ liệu: {e}")
        return

    # Lọc trọng số chỉ lấy các cột có trong DataFrame để đảm bảo thứ tự khớp
    weights_series = pd.Series(criteria_weights).loc[df_data.columns]

    # --- CÁC BƯỚC TOPSIS TIẾP THEO GIỮ NGUYÊN (1-6) ---

    # --- BƯỚC 1: CHUẨN HÓA MA TRẬN (VECTOR NORMALIZATION) ---
    sum_of_squares = (df_data ** 2).sum() ** 0.5
    sum_of_squares.replace(0, 1e-6, inplace=True)  # Thay đổi để tránh chia cho 0
    df_norm = df_data / sum_of_squares

    # --- BƯỚC 2: TÍNH MA TRẬN CHUẨN HÓA CÓ TRỌNG SỐ ---
    df_weighted = df_norm * weights_series

    # --- BƯỚC 3: XÁC ĐỊNH GIẢI PHÁP LÝ TƯỞNG TỐT NHẤT (A+) VÀ TỆ NHẤT (A-) ---
    ideal_best = {}
    ideal_worst = {}

    for col in df_weighted.columns:
        # Chỉ lấy loại tiêu chí từ metadata (criteria_types)
        c_type = criteria_types.get(col)

        if c_type == 'benefit':
            ideal_best[col] = df_weighted[col].max()
            ideal_worst[col] = df_weighted[col].min()
        elif c_type == 'cost':
            ideal_best[col] = df_weighted[col].min()
            ideal_worst[col] = df_weighted[col].max()

        # --- BƯỚC 4: TÍNH KHOẢNG CÁCH EUCLIDEAN (S+ VÀ S-) ---

    ideal_best_series = pd.Series(ideal_best)
    ideal_worst_series = pd.Series(ideal_worst)

    dist_to_best = np.sqrt(((df_weighted - ideal_best_series) ** 2).sum(axis=1))
    dist_to_worst = np.sqrt(((df_weighted - ideal_worst_series) ** 2).sum(axis=1))

    # --- BƯỚC 5: TÍNH CHỈ SỐ TƯƠNG ĐỐI (C_i*) ---

    total_distance = dist_to_best + dist_to_worst
    total_distance.replace(0, 1e-6, inplace=True)

    closeness_score = dist_to_worst / total_distance

    # --- BƯỚC 6: XẾP HẠNG VÀ BÁO CÁO ---

    df_report = df_locations[['ward']].copy()  # Dùng 'Ward'
    df_report['TOPSIS_Score'] = closeness_score.values  # Dùng .values để gán đúng

    df_report = df_report.sort_values(by='TOPSIS_Score', ascending=False)
    df_report['Rank'] = range(1, len(df_report) + 1)

    # Đổi tên cột hiển thị
    report_df = df_report.rename(
        columns={'ward': 'Tên phường', 'TOPSIS_Score': 'Điểm TOPSIS (0-1)'})
    return report_df


# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":

    # BƯỚC MỚI: Tải tất cả trọng số tiêu chí từ YAML
    ALL_AHP_WEIGHTS = load_criteria_weights(CRITERIA_WEIGHTS_PATH)

    if ALL_AHP_WEIGHTS is not None:
        # BƯỚC 1: Lấy lựa chọn mô hình từ người dùng
        selected_type = get_user_selection(ALL_AHP_WEIGHTS)

        # BƯỚC 2: Chạy phân tích TOPSIS
        if selected_type:
            run_topsis_model(CSV_PATH, JSON_PATH, selected_type, ALL_AHP_WEIGHTS)