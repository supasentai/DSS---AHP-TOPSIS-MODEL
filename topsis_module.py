import os
import json
import numpy as np
import pandas as pd


def load_metadata(json_path: str) -> dict | None:
    """
    Đọc metadata và lấy loại tiêu chí (benefit/cost).

    Parameters
    ----------
    json_path : str
        Đường dẫn file metadata.json.

    Returns
    -------
    dict | None
        {tên_tiêu_chí: 'benefit' | 'cost'} hoặc None nếu lỗi.
    """
    try:
        with open(json_path, "r", encoding="utf-8-sig") as f:
            meta = json.load(f)
        return {k: v.get("type") for k, v in meta.items() if v.get("type") in ("benefit", "cost")}
    except Exception as e:
        print(f"Lỗi khi tải metadata: {e}")
        return None


def run_topsis_model(
    data_path: str,
    json_path: str,
    analysis_type: str,
    all_criteria_weights: dict,
) -> pd.DataFrame | None:
    """
    Chạy TOPSIS cho `analysis_type` với bộ trọng số trong `all_criteria_weights`.
    Trả về DataFrame kết quả và lưu ra `data/ranking_result_{analysis_type}.xlsx`
    (trừ khi tên mô hình chứa 'temp').

    Parameters
    ----------
    data_path : str
        Đường dẫn file dữ liệu (Excel).
    json_path : str
        Đường dẫn metadata.json.
    analysis_type : str
        Tên mô hình trọng số cần chạy.
    all_criteria_weights : dict
        Mapping {model_name: {criterion: weight}}.

    Returns
    -------
    pd.DataFrame | None
        Bảng xếp hạng gồm: Tên phường, Điểm TOPSIS (0-1), Rank.
    """
    print(f"\n--- CHẠY TOPSIS: {analysis_type.upper()} ---")

    weights = all_criteria_weights.get(analysis_type)
    if not weights:
        print(f"Lỗi: Không tìm thấy trọng số cho '{analysis_type}'.")
        return None

    crit_types = load_metadata(json_path)
    if crit_types is None:
        return None

    try:
        df = pd.read_excel(data_path)
        # Giữ tiêu chí có trong metadata + trọng số + cột dữ liệu
        valid_criteria = [c for c in crit_types if c in weights and c in df.columns]
        weights = {k: v for k, v in weights.items() if k in valid_criteria}

        df_locations = df[["Tỉnh/Thành phố"]].copy()
        df_data = df[valid_criteria].astype(float)
    except Exception as e:
        print(f"Lỗi khi đọc dữ liệu: {e}")
        return None

    # Bảo toàn thứ tự cột khi nhân trọng số
    w_series = pd.Series(weights).loc[df_data.columns]

    # 1) Chuẩn hóa vector theo cột
    denom = (df_data ** 2).sum() ** 0.5
    denom.replace(0, 1e-6, inplace=True)
    df_norm = df_data / denom

    # 2) Áp trọng số
    df_weighted = df_norm * w_series

    # 3) Ideal best/worst theo loại tiêu chí
    ideal_best, ideal_worst = {}, {}
    for col in df_weighted.columns:
        t = crit_types.get(col)
        if t == "benefit":
            ideal_best[col] = df_weighted[col].max()
            ideal_worst[col] = df_weighted[col].min()
        elif t == "cost":
            ideal_best[col] = df_weighted[col].min()
            ideal_worst[col] = df_weighted[col].max()

    # 4) Khoảng cách tới ideal
    ib = pd.Series(ideal_best)
    iw = pd.Series(ideal_worst)
    d_best = np.sqrt(((df_weighted - ib) ** 2).sum(axis=1))
    d_wrst = np.sqrt(((df_weighted - iw) ** 2).sum(axis=1))

    # 5) Điểm gần lý tưởng
    denom = d_best + d_wrst
    denom.replace(0, 1e-6, inplace=True)
    score = d_wrst / denom

    # 6) Xếp hạng và báo cáo
    out = df_locations.copy()
    out["TOPSIS_Score"] = score.values
    out = out.sort_values(by="TOPSIS_Score", ascending=False)
    out["Rank"] = range(1, len(out) + 1)

    report_df = out.rename(columns={"Tỉnh/Thành phố": "Tên Tỉnh/Thành", "TOPSIS_Score": "Điểm TOPSIS (0-1)"})

    # Lưu Excel nếu không phải mô hình tạm
    if "temp" not in analysis_type:
        fn = f"data/ranking_result_{analysis_type}.xlsx"
        try:
            report_df.to_excel(fn, index=False)  # giữ nguyên hành vi: mặc định có index
            print(f"Đã lưu: {fn}")
        except Exception as e:
            print(f"Lỗi khi lưu file: {e}")

    return report_df
