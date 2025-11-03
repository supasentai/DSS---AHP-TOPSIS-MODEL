import pandas as pd
import numpy as np
import json
import os
import yaml


def load_metadata(json_path):
    """Tải metadata và trích xuất loại tiêu chí (benefit/cost)."""
    try:
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            metadata = json.load(f)
        return {k: v.get('type') for k, v in metadata.items() if v.get('type') in ['benefit', 'cost']}
    except Exception as e:
        print(f"Lỗi khi tải metadata: {e}")
        return None


def run_topsis_model(data_path, json_path, analysis_type, all_criteria_weights):
    """
    Hàm chính TOPSIS, sẽ return DataFrame VÀ lưu kết quả ra file CSV.
    """
    print(f"\n--- KHỞI CHẠY MÔ HINH TOPSIS CHO: {analysis_type.upper()} ---")

    criteria_weights = all_criteria_weights.get(analysis_type)
    if not criteria_weights:
        print(f"LỖI: Không tìm thấy trọng số cho '{analysis_type}'.")
        return None

    criteria_types = load_metadata(json_path)
    if criteria_types is None:
        return None

    try:
        df = pd.read_excel(data_path)
        # Chỉ giữ lại các tiêu chí có trong metadata VÀ trọng số
        valid_criteria = [c for c in criteria_types.keys() if c in criteria_weights and c in df.columns]

        # Lọc lại criteria_weights chỉ với các tiêu chí hợp lệ
        criteria_weights = {k: v for k, v in criteria_weights.items() if k in valid_criteria}

        df_locations = df[['ward']].copy()
        df_data = df[valid_criteria].astype(float)
    except Exception as e:
        print(f"Lỗi khi đọc file CSV hoặc xử lý dữ liệu: {e}")
        return None

    weights_series = pd.Series(criteria_weights).loc[df_data.columns]

    # --- BƯỚC 1: CHUẨN HÓA ---
    sum_of_squares = (df_data ** 2).sum() ** 0.5
    sum_of_squares.replace(0, 1e-6, inplace=True)
    df_norm = df_data / sum_of_squares

    # --- BƯỚC 2: NHÂN TRỌNG SỐ ---
    df_weighted = df_norm * weights_series

    # --- BƯỚC 3: TÌM GIẢI PHÁP LÝ TƯỞNG ---
    ideal_best = {}
    ideal_worst = {}
    for col in df_weighted.columns:
        c_type = criteria_types.get(col)
        if c_type == 'benefit':
            ideal_best[col] = df_weighted[col].max()
            ideal_worst[col] = df_weighted[col].min()
        elif c_type == 'cost':
            ideal_best[col] = df_weighted[col].min()
            ideal_worst[col] = df_weighted[col].max()

    # --- BƯỚC 4: TÍNH KHOẢNG CÁCH ---
    ideal_best_series = pd.Series(ideal_best)
    ideal_worst_series = pd.Series(ideal_worst)
    dist_to_best = np.sqrt(((df_weighted - ideal_best_series) ** 2).sum(axis=1))
    dist_to_worst = np.sqrt(((df_weighted - ideal_worst_series) ** 2).sum(axis=1))

    # --- BƯỚC 5: TÍNH ĐIỂM TƯƠNG ĐỐI ---
    total_distance = dist_to_best + dist_to_worst
    total_distance.replace(0, 1e-6, inplace=True)
    closeness_score = dist_to_worst / total_distance

    # --- BƯỚC 6: XẾP HẠNG VÀ BÁO CÁO ---
    df_report = df_locations.copy()
    df_report['TOPSIS_Score'] = closeness_score.values
    df_report = df_report.sort_values(by='TOPSIS_Score', ascending=False)
    df_report['Rank'] = range(1, len(df_report) + 1)

    report_df = df_report.rename(columns={'ward': 'Tên phường', 'TOPSIS_Score': 'Điểm TOPSIS (0-1)'})

    # --- CẬP NHẬT MỚI: LƯU KẾT QUẢ RA FILE ---
    # Chỉ lưu file nếu tên mô hình không phải là tên tạm thời (what-if)
    if "temp" not in analysis_type:
        output_filename = f"data/ranking_result_{analysis_type}.xlsx"
        try:
            # Dùng utf-8-sig để Excel đọc CSV tiếng Việt không bị lỗi
            report_df.to_excel(output_filename)
            print(f"Đã lưu kết quả vào file: {output_filename}")
        except Exception as e:
            print(f"Lỗi khi lưu file: {e}")  # Vẫn tiếp tục chạy dù lưu file lỗi

    # Trả về DataFrame để Streamlit sử dụng
    return report_df

