import os
import json
import numpy as np
import pandas as pd


def load_metadata(json_path: str) -> dict:
    """
    Đọc đầy đủ metadata từ JSON.

    Trả về dict với key là tên tiêu chí trong metadata (ví dụ: "criteria_1")
    và value là dict con chứa type, display_name, v.v.
    """
    try:
        with open(json_path, "r", encoding="utf-8-sig") as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) else {}
    except Exception as e:
        print(f"Lỗi khi tải metadata: {e}")
        return {}


def _canonical_key(name: str) -> str:
    """Chuẩn hoá tên cột: 'Criteria 1' -> 'criteria_1'."""
    s = str(name).strip()
    return s.lower().replace(" ", "_")


def _criterion_type(col_name: str, meta: dict) -> str | None:
    """
    Lấy kiểu tiêu chí (benefit/cost) cho một cột dữ liệu.

    Ưu tiên khớp key đúng, nếu không thì thử dạng canonical ('criteria_1').
    """
    # khớp trực tiếp
    info = meta.get(col_name)
    if isinstance(info, dict):
        t = info.get("type")
        if t in ("benefit", "cost"):
            return t

    # khớp canonical
    alt = _canonical_key(col_name)
    info = meta.get(alt)
    if isinstance(info, dict):
        t = info.get("type")
        if t in ("benefit", "cost"):
            return t

    return None


def run_topsis_model(
    data_path: str,
    json_path: str,
    analysis_type: str,
    all_criteria_weights: dict,
) -> pd.DataFrame | None:
    """
    Chạy TOPSIS cho `analysis_type` với bộ trọng số trong `all_criteria_weights`.

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
        Bảng xếp hạng gồm: Tên Tỉnh/Thành (hoặc ID tỉnh), Điểm TOPSIS (0-1), Rank.
    """
    print(f"\n--- CHẠY TOPSIS: {analysis_type.upper()} ---")

    # 1. Lấy trọng số cho mô hình
    weights_raw = all_criteria_weights.get(analysis_type)
    if not weights_raw:
        print(f"Lỗi: Không tìm thấy trọng số cho '{analysis_type}'.")
        return None

    # convert về float và string key
    weights_raw = {str(k): float(v) for k, v in weights_raw.items()}

    # 2. Lấy metadata (để biết benefit / cost)
    meta = load_metadata(json_path)

    # 3. Đọc dữ liệu Excel và ghép tên tỉnh, vùng (giống logic trong app.py)
    try:
        xls = pd.ExcelFile(data_path)
        df_main = xls.parse("Sheet1")  # province_id + criteria_x + region_id
        df = df_main.copy()

        # Ghép tên Tỉnh/Thành từ sheet "Tổng hợp" (nếu có)
        try:
            df_info = xls.parse("Tổng hợp")
            if "province_id" in df_info.columns and "Tỉnh/Thành phố" in df_info.columns:
                df_info = df_info[["province_id", "Tỉnh/Thành phố"]].drop_duplicates(subset=["province_id"])
                df = df.merge(df_info, on="province_id", how="left")
        except Exception:
            pass

        # Ghép vùng từ sheet "Vùng" (nếu có)
        try:
            df_region = xls.parse("Vùng")
            if "province_id" in df_region.columns and "region" in df_region.columns:
                df_region = df_region[["province_id", "region"]].drop_duplicates(subset=["province_id"])
                df = df.merge(df_region, on="province_id", how="left")
        except Exception:
            pass

        if "Tỉnh/Thành phố" not in df.columns:
            raise KeyError("Thiếu cột 'Tỉnh/Thành phố' sau khi ghép dữ liệu.")
    except Exception as e:
        print(f"Lỗi khi đọc dữ liệu: {e}")
        return None

    # 4. Chọn các tiêu chí hợp lệ dựa trên giao giữa cột dữ liệu & trọng số,
    #    đồng thời suy ra loại benefit/cost từ metadata.
    valid_criteria: list[str] = []
    crit_type_map: dict[str, str] = {}

    for col in weights_raw.keys():
        if col not in df.columns:
            continue
        t = _criterion_type(col, meta)
        if t not in ("benefit", "cost"):
            continue
        valid_criteria.append(col)
        crit_type_map[col] = t

    if not valid_criteria:
        print("Lỗi: Không tìm thấy tiêu chí chung giữa metadata, trọng số và dữ liệu.")
        return None

    # Lọc weights và dữ liệu cho đúng các cột thực tế
    weights = {k: float(weights_raw[k]) for k in valid_criteria}
    try:
        df_locations = df[["Tỉnh/Thành phố"]].copy()
        df_data = df[valid_criteria].astype(float)
    except Exception as e:
        print(f"Lỗi khi chuẩn bị dữ liệu tiêu chí: {e}")
        return None

    # 5. Chuẩn hóa và nhân trọng số
    w_series = pd.Series(weights, index=valid_criteria)
    if w_series.sum() == 0:
        print("Lỗi: Tổng trọng số bằng 0.")
        return None
    w_series = w_series / w_series.sum()

    # Chuẩn hóa vector theo cột (Euclid)
    denom = (df_data ** 2).sum() ** 0.5
    denom.replace(0, 1e-6, inplace=True)
    df_norm = df_data / denom

    # Áp trọng số
    df_weighted = df_norm * w_series

    # 6. Xác định phương án tốt nhất/xấu nhất
    crit_type_series = pd.Series(crit_type_map)
    ideal_best = pd.Series(index=valid_criteria, dtype=float)
    ideal_worst = pd.Series(index=valid_criteria, dtype=float)

    for c in valid_criteria:
        col = df_weighted[c]
        if crit_type_series[c] == "benefit":
            ideal_best[c] = col.max()
            ideal_worst[c] = col.min()
        else:  # cost
            ideal_best[c] = col.min()
            ideal_worst[c] = col.max()

    # 7. Tính khoảng cách và điểm TOPSIS
    dist_best = np.sqrt(((df_weighted - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((df_weighted - ideal_worst) ** 2).sum(axis=1))

    denom_dist = dist_best + dist_worst
    denom_dist[denom_dist == 0] = 1e-6
    topsis_score = dist_worst / denom_dist

    # 8. Ghép kết quả, sắp xếp và đánh rank
    out = df_locations.copy()
    out["TOPSIS_Score"] = topsis_score
    out = out.sort_values("TOPSIS_Score", ascending=False).reset_index(drop=True)
    out["Rank"] = range(1, len(out) + 1)

    report_df = out.rename(
        columns={
            "Tỉnh/Thành phố": "Tên Tỉnh/Thành",
            "TOPSIS_Score": "Điểm TOPSIS (0-1)",
        }
    )

    # 9. Lưu Excel nếu không phải mô hình tạm
    if "temp" not in analysis_type:
        fn = os.path.join("data", f"ranking_result_{analysis_type}.xlsx")
        try:
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            report_df.to_excel(fn, index=False)
            print(f"Đã lưu: {fn}")
        except Exception as e:
            print(f"Lỗi khi lưu file: {e}")

    return report_df
