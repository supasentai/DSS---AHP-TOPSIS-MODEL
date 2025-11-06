import os
import pandas as pd
from topsis_module import run_topsis_model

# Đường dẫn dữ liệu
EX_PATH = "data/AHP_Data_synced_fixed.xlsx"
JSON_PATH = "data/metadata.json"


def load_original_ranking(model_name: str, all_weights: dict) -> pd.DataFrame | None:
    """
    Tải bảng xếp hạng gốc cho model_name.
    Nếu file chưa tồn tại, chạy TOPSIS lần đầu (dùng all_weights) để tạo.

    Trả về:
        DataFrame xếp hạng hoặc None nếu lỗi.
    """
    filename = f"data/ranking_result_{model_name}.xlsx"
    if os.path.exists(filename):
        try:
            return pd.read_excel(filename)
        except Exception as e:
            print(f"Lỗi đọc '{filename}': {e}")
            return None

    # Chưa có file → chạy TOPSIS để tạo
    try:
        return run_topsis_model(EX_PATH, JSON_PATH, model_name, all_weights)
    except Exception as e:
        print(f"Lỗi khi tạo ranking gốc bằng TOPSIS: {e}")
        return None


def run_what_if_analysis(original_model_name: str,
                         new_weights_dict: dict,
                         all_weights: dict) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Phân tích độ nhạy:
      1) Lấy xếp hạng gốc của original_model_name.
      2) Chạy TOPSIS với trọng số mới (new_weights_dict) dưới tên tạm 'what_if_temp'.
      3) Trả về (original_df, new_df).

    Trả về:
        (DataFrame gốc, DataFrame mới) hoặc (None, None) nếu lỗi.
    """
    original_ranking = load_original_ranking(original_model_name, all_weights)
    if original_ranking is None:
        return None, None

    what_if_model_name = "what_if_temp"
    all_what_if_weights = {what_if_model_name: new_weights_dict}

    try:
        new_ranking = run_topsis_model(EX_PATH, JSON_PATH, what_if_model_name, all_what_if_weights)
        return original_ranking, new_ranking
    except Exception as e:
        print(f"Lỗi khi chạy TOPSIS What-if: {e}")
        return None, None
