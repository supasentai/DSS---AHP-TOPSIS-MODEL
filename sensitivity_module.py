import pandas as pd
import yaml
import os
from topsis_module import run_topsis_model # Import hàm TOPSIS

# --- CẤU HÌNH ĐƯỜNG DẪN ---
EX_PATH = "data/AHP_Data_synced_fixed.xlsx"
JSON_PATH = "data/metadata.json"
# ### SỬA ĐỔI: Phân biệt default và user weights ###
DEFAULT_WEIGHTS_PATH = "data/defaultweights.yaml"
USER_WEIGHTS_PATH = "data/weights.yaml"
CRITERIA_WEIGHTS_PATH = USER_WEIGHTS_PATH

def load_original_ranking(model_name):
    """
    Tải file CSV chứa kết quả xếp hạng gốc.
    Nếu file không tồn tại, chạy TOPSIS để tạo ra nó.
    """
    filename = f"ranking_result_{model_name}.xlsx"
    if os.path.exists(filename):
        try:
            return pd.read_excel(filename)
        except Exception as e:
            print(f"Lỗi khi đọc file ranking gốc: {e}")
            return None
    else:
        # File chưa tồn tại, chạy lần đầu để tạo ra nó
        print(f"File {filename} chưa có. Chạy TOPSIS lần đầu để tạo...")
        try:
            all_weights = {}
            if os.path.exists(DEFAULT_WEIGHTS_PATH):
                with open(DEFAULT_WEIGHTS_PATH, 'r', encoding='utf-8') as f:
                    all_weights.update(yaml.safe_load(f) or {})
            if os.path.exists(USER_WEIGHTS_PATH):
                with open(USER_WEIGHTS_PATH, 'r', encoding='utf-8') as f:
                    all_weights.update(yaml.safe_load(f) or {})

            if not all_weights:
                print(f"LỖI: Không tìm thấy file trọng số nào ({DEFAULT_WEIGHTS_PATH} hoặc {USER_WEIGHTS_PATH}).")
                return None
            # Hàm run_topsis_model sẽ tự động lưu file
            original_df = run_topsis_model(EX_PATH, JSON_PATH, model_name, all_weights)
            return original_df
        except Exception as e:
            print(f"Lỗi khi tạo file ranking gốc: {e}")
            return None

def run_what_if_analysis(original_model_name, new_weights_dict):
    """
    Chạy phân tích độ nhạy:
    1. Tải kết quả gốc.
    2. Chạy TOPSIS với trọng số mới.
    3. Trả về cả hai DataFrame.
    """
    # 1. Tải bảng xếp hạng gốc
    original_ranking = load_original_ranking(original_model_name)
    if original_ranking is None:
        return None, None

    # 2. Chuẩn bị dữ liệu cho chạy mới
    what_if_model_name = "what_if_temp" # Tên tạm thời
    all_what_if_weights = {what_if_model_name: new_weights_dict}

    # 3. Chạy TOPSIS với trọng số mới
    new_ranking = run_topsis_model(
        EX_PATH,
        JSON_PATH,
        what_if_model_name,
        all_what_if_weights
    )

    return original_ranking, new_ranking

