"""
Phiên bản Console (CLI) của ứng dụng DSS Quận 7.
Chạy logic nghiệp vụ không cần Streamlit. Hỗ trợ:
1) Tổng quan dữ liệu
2) Quản lý trọng số AHP (direct rating, pairwise)
3) Chạy TOPSIS
4) Phân tích What-if
5) Xuất báo cáo PDF
"""

import os
import sys
import json
import yaml
import numpy as np
import pandas as pd

# =========================
# Đường dẫn
# =========================
DATA_PATH = "data/AHP_Data_synced_fixed.xlsx"
METADATA_PATH = "data/metadata.json"
WEIGHTS_PATH = "data/weights.yaml"             # chỉ dùng 1 file weights
DEFAULT_WEIGHTS_PATH = "data/defaultweights.yaml"

# =========================
# Import module nghiệp vụ
# =========================
try:
    from ahp_module import calculate_ahp_weights, save_weights_to_yaml
    from topsis_module import run_topsis_model
    from sensitivity_module import run_what_if_analysis
    from report_module import create_full_report
except ImportError as e:
    print("LỖI: Không thể import module. Hãy đặt các file *.py cùng thư mục.")
    print("Thiếu: ahp_module.py, topsis_module.py, sensitivity_module.py, report_module.py")
    print(f"Chi tiết: {e}")
    sys.exit(1)

# =========================
# Helpers giao diện console
# =========================
def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")

def wait_for_enter() -> None:
    input("\n... Nhấn Enter để tiếp tục ...")

def nice_name(col: str) -> str:
    return str(col).replace("_", " ").strip().title()

def get_all_criteria() -> list[str]:
    """Tải danh sách tất cả tiêu chí từ file dữ liệu."""
    try:
        df = pd.read_excel(DATA_PATH)
        return [c for c in df.columns if c not in ["ward", "ward_id"]]
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file dữ liệu: {DATA_PATH}")
        return []
    except Exception as e:
        print(f"LỖI khi đọc dữ liệu: {e}")
        return []

# =========================
# Weights I/O
# =========================
def load_weights_file() -> dict:
    """
    Tải weights.yaml. Nếu chưa có thì khởi tạo từ defaultweights.yaml (nếu tồn tại).
    """
    if not os.path.exists(WEIGHTS_PATH):
        if os.path.exists(DEFAULT_WEIGHTS_PATH):
            print(f"Khởi tạo {WEIGHTS_PATH} từ {DEFAULT_WEIGHTS_PATH} ...")
            try:
                with open(DEFAULT_WEIGHTS_PATH, "r", encoding="utf-8") as f0:
                    default_data = yaml.safe_load(f0)
                with open(WEIGHTS_PATH, "w", encoding="utf-8") as f1:
                    yaml.dump(default_data, f1, allow_unicode=True, sort_keys=False, indent=2)
                return default_data or {}
            except Exception as e:
                print(f"Lỗi khi khởi tạo {WEIGHTS_PATH}: {e}")
                return {}
        else:
            print(f"Cảnh báo: Không có {WEIGHTS_PATH} và {DEFAULT_WEIGHTS_PATH}. Bắt đầu rỗng.")
            return {}
    try:
        with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"LỖI khi đọc {WEIGHTS_PATH}: {e}")
        return {}

def select_model(all_weights: dict) -> tuple[str | None, dict | None]:
    """Cho người dùng chọn một mô hình AHP đã lưu."""
    if not all_weights:
        print("Chưa có mô hình AHP nào.")
        return None, None

    models = list(all_weights.keys())
    print("--- Chọn một mô hình ---")
    for i, name in enumerate(models, start=1):
        print(f" [{i}] {name}")
    print(" [0] Quay lại")

    while True:
        try:
            choice = int(input("Nhập lựa chọn: ").strip())
            if 0 < choice <= len(models):
                m = models[choice - 1]
                return m, all_weights[m]
            if choice == 0:
                return None, None
            print("Lựa chọn không hợp lệ.")
        except ValueError:
            print("Vui lòng nhập số.")

# =========================
# Chức năng 1: Tổng quan dữ liệu
# =========================
def show_data_overview() -> None:
    clear_screen()
    print("===================================")
    print("  1. TỔNG QUAN DỮ LIỆU")
    print("===================================")
    try:
        df = pd.read_excel(DATA_PATH)
        print("\n--- Thống kê mô tả (5 hàng đầu) ---")
        print(df.describe().T.head())
        print("\n--- Dữ liệu thô (5 hàng đầu) ---")
        print(df.head())
        print(f"\nTổng cộng: {len(df)} phường, {len(get_all_criteria())} tiêu chí.")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file dữ liệu: {DATA_PATH}")
    except Exception as e:
        print(f"LỖI khi đọc dữ liệu: {e}")
    wait_for_enter()

# =========================
# Chức năng 2: Quản lý AHP
# =========================
def manage_ahp() -> None:
    clear_screen()
    print("===================================")
    print("  2. QUẢN LÝ TRỌNG SỐ (AHP)")
    print("===================================")

    all_weights = load_weights_file()
    print("Các mô hình hiện có:", list(all_weights.keys()) or "[Trống]")
    print("\n [1] Tạo/Tùy chỉnh — Điểm 1-10")
    print(" [2] Tạo/Tùy chỉnh — Ma trận so sánh cặp")
    print(" [0] Quay lại Menu")

    choice = input("Nhập lựa chọn: ").strip()
    if choice == "1":
        ahp_method_1(all_weights)
    elif choice == "2":
        ahp_method_2(all_weights)

def ahp_method_1(all_weights: dict) -> None:
    """AHP kiểu direct rating 1–10."""
    print("\n--- AHP: Tạo mới (Điểm 1–10) ---")
    model_name = input("Tên mô hình (nếu trùng sẽ tự đổi): ").strip()
    if not model_name:
        print("Tên mô hình không được rỗng.")
        wait_for_enter()
        return

    all_criteria = get_all_criteria()
    if not all_criteria:
        return

    original = all_weights.get(model_name, {})
    default_scores: dict[str, int] = {}
    if original:
        max_w = max(original.values()) or 1
        default_scores = {k: int(round((v / max_w) * 9 + 1)) for k, v in original.items()}

    print("\nChọn tiêu chí (ví dụ: 1,3,5). Bỏ trống = mặc định.")
    default_select = list(original.keys()) if original else all_criteria
    for i, c in enumerate(all_criteria, start=1):
        mark = " (*)" if c in default_select else ""
        print(f" [{i}] {nice_name(c)}{mark}")

    try:
        raw = input("Lựa chọn: ").strip()
        if not raw:
            selected = default_select
        else:
            idx = [int(x.strip()) - 1 for x in raw.split(",")]
            selected = [all_criteria[i] for i in idx if 0 <= i < len(all_criteria)]
    except ValueError:
        print("Lỗi: nhập không hợp lệ.")
        wait_for_enter()
        return

    if not selected:
        print("Chưa chọn tiêu chí.")
        wait_for_enter()
        return

    print("\nNhập điểm 1–10 cho từng tiêu chí:")
    scores: dict[str, float] = {}
    for crit in selected:
        while True:
            try:
                dft = default_scores.get(crit, 5)
                s = input(f"  {nice_name(crit)} (mặc định {dft}): ").strip()
                score = dft if not s else float(s)
                if 1 <= score <= 10:
                    scores[crit] = score
                    break
                print("Điểm phải từ 1 đến 10.")
            except ValueError:
                print("Vui lòng nhập số.")

    total = sum(scores.values())
    if total == 0:
        print("Lỗi: Tổng điểm bằng 0.")
        wait_for_enter()
        return

    weights_norm = {k: v / total for k, v in scores.items()}
    print("\n--- Trọng số đã chuẩn hóa ---")
    print(json.dumps(weights_norm, indent=2, ensure_ascii=False))

    ok, final_name = save_weights_to_yaml(weights_norm, model_name, WEIGHTS_PATH)
    if ok:
        if final_name != model_name:
            print(f"\nĐã lưu với tên mới do trùng: '{final_name}'")
        else:
            print(f"\nĐã lưu mô hình '{final_name}' vào {WEIGHTS_PATH}.")
    else:
        print("\nLỖI: Không thể lưu mô hình.")
    wait_for_enter()

def ahp_method_2(all_weights: dict) -> None:
    """AHP kiểu pairwise matrix (Saaty 1/9..9)."""
    print("\n--- AHP: Tạo mới (Ma trận so sánh cặp) ---")
    model_name = input("Tên mô hình (nếu trùng sẽ tự đổi): ").strip()
    if not model_name:
        print("Tên mô hình không được rỗng.")
        wait_for_enter()
        return

    all_criteria = get_all_criteria()
    if not all_criteria:
        return

    original = all_weights.get(model_name, {})
    default_select = list(original.keys()) if original else all_criteria

    print("\nChọn tiêu chí (ví dụ: 1,3,5). Bỏ trống = mặc định.")
    for i, c in enumerate(all_criteria, start=1):
        mark = " (*)" if c in default_select else ""
        print(f" [{i}] {nice_name(c)}{mark}")

    try:
        raw = input("Lựa chọn: ").strip()
        if not raw:
            selected = default_select
        else:
            idx = [int(x.strip()) - 1 for x in raw.split(",")]
            selected = [all_criteria[i] for i in idx if 0 <= i < len(all_criteria)]
    except ValueError:
        print("Lỗi: nhập không hợp lệ.")
        wait_for_enter()
        return

    n = len(selected)
    if n < 2:
        print("Cần ít nhất 2 tiêu chí.")
        wait_for_enter()
        return

    A = np.ones((n, n), dtype=float)
    print("\nNhập so sánh cặp (1/9 .. 9). Ví dụ 5 nghĩa là hàng i quan trọng hơn cột j mức 5.")
    for i in range(n):
        for j in range(i + 1, n):
            while True:
                try:
                    val = float(input(f"  {nice_name(selected[i])} vs {nice_name(selected[j])}: ").strip())
                    if 1/9 <= val <= 9:
                        A[i, j] = val
                        A[j, i] = 1.0 / val
                        break
                    print("Giá trị phải trong [1/9, 9].")
                except ValueError:
                    print("Vui lòng nhập số.")

    weights, cr = calculate_ahp_weights(A)
    if weights is None:
        print("\nLỖI: Không tính được AHP.")
        wait_for_enter()
        return

    print("\n--- Kết quả ---")
    if cr is not None:
        print(f"  Consistency Ratio (CR): {cr:.4f}")
        if cr >= 0.1:
            print("  CẢNH BÁO: CR ≥ 0.1. Nên xem xét lại ma trận.")
        else:
            print("  CR < 0.1. Chấp nhận được.")
    else:
        print("  Không có RI phù hợp kích thước. Bỏ qua kiểm tra nhất quán.")

    weights_dict = {name: float(w) for name, w in zip(selected, weights)}
    print("\n--- Trọng số ---")
    print(json.dumps(weights_dict, indent=2, ensure_ascii=False))

    ok, final_name = save_weights_to_yaml(weights_dict, model_name, WEIGHTS_PATH)
    if ok:
        if final_name != model_name:
            print(f"\nĐã lưu với tên mới do trùng: '{final_name}'")
        else:
            print(f"\nĐã lưu mô hình '{final_name}' vào {WEIGHTS_PATH}.")
    else:
        print("\nLỖI: Không thể lưu mô hình.")
    wait_for_enter()

# =========================
# Chức năng 3: Chạy TOPSIS
# =========================
def run_topsis_console() -> None:
    clear_screen()
    print("===================================")
    print("  3. CHẠY XẾP HẠNG (TOPSIS)")
    print("===================================")

    all_weights = load_weights_file()
    model_name, _ = select_model(all_weights)
    if not model_name:
        return

    print(f"\nĐang chạy TOPSIS cho mô hình: {model_name} ...")
    report_df = run_topsis_model(
        data_path=DATA_PATH,
        json_path=METADATA_PATH,
        analysis_type=model_name,
        all_criteria_weights=all_weights,
    )
    if report_df is not None:
        print("\n--- KẾT QUẢ XẾP HẠNG TOPSIS ---")
        print(report_df)
        print(f"\nĐã lưu: data/ranking_result_{model_name}.xlsx")
    else:
        print("LỖI: Chạy TOPSIS thất bại.")
    wait_for_enter()

# =========================
# Chức năng 4: What-if
# =========================
def run_what_if_console() -> None:
    clear_screen()
    print("===================================")
    print("  4. CHẠY PHÂN TÍCH (WHAT-IF)")
    print("===================================")

    all_weights = load_weights_file()
    print("Chọn mô hình GỐC để chạy What-if:")
    model_name, original = select_model(all_weights)
    if not model_name:
        return

    print(f"\n--- Điều chỉnh trọng số (dựa trên '{model_name}') ---")
    print("Nhấn Enter để giữ nguyên giá trị cũ.")
    new_w: dict[str, float] = {}
    for crit, w_old in original.items():
        while True:
            try:
                s = input(f"  {nice_name(crit)} (cũ: {w_old:.4f}): ").strip()
                new_w[crit] = w_old if not s else float(s)
                break
            except ValueError:
                print("Vui lòng nhập số.")

    total = sum(new_w.values())
    normalized = {k: (v / total if total > 0 else 0.0) for k, v in new_w.items()}

    print("\nĐang chạy What-if ...")
    orig_df, new_df = run_what_if_analysis(model_name, normalized, all_weights)
    if orig_df is not None and new_df is not None:
        print("\n--- BẢNG XẾP HẠNG GỐC ---")
        print(orig_df)
        print("\n--- BẢNG XẾP HẠNG MỚI (WHAT-IF) ---")
        print(new_df)

        df_o = orig_df[["Tên phường", "Rank"]].rename(columns={"Rank": "Hạng Gốc"})
        df_n = new_df[["Tên phường", "Rank"]].rename(columns={"Rank": "Hạng Mới"})
        comp = pd.merge(df_o, df_n, on="Tên phường")
        comp["Thay đổi"] = comp["Hạng Gốc"] - comp["Hạng Mới"]
        print("\n--- SO SÁNH THAY ĐỔI (Gốc - Mới) ---")
        print(comp.sort_values(by="Hạng Mới"))
    else:
        print("LỖI: Chạy What-if thất bại.")
    wait_for_enter()

# =========================
# Chức năng 5: PDF Report
# =========================
def export_report_console() -> None:
    clear_screen()
    print("===================================")
    print("  5. XUẤT BÁO CÁO PDF (TEST)")
    print("===================================")
    print("Yêu cầu: đã có mô hình AHP và file kết quả TOPSIS tương ứng.")

    all_weights = load_weights_file()
    model_name, _ = select_model(all_weights)
    if not model_name:
        return

    ranking_file = f"data/ranking_result_{model_name}.xlsx"
    if not os.path.exists(ranking_file):
        print(f"\nLỖI: Chưa thấy '{ranking_file}'. Hãy chạy TOPSIS trước.")
        wait_for_enter()
        return

    print(f"\nĐang tạo báo cáo PDF cho mô hình: {model_name} ...")
    try:
        pdf_path = create_full_report(model_name, all_weights)
        if pdf_path:
            print("\n--- THÀNH CÔNG ---")
            print(f"Đã lưu báo cáo: {pdf_path}")
            try:
                os.startfile(pdf_path)  # Windows
            except AttributeError:
                try:
                    os.system(f'open "{pdf_path}"')  # macOS
                except Exception:
                    os.system(f'xdg-open "{pdf_path}"')  # Linux
        else:
            print("\nLỖI: Không thể tạo báo cáo PDF.")
    except Exception as e:
        print(f"\nLỖI ngoại lệ khi tạo PDF: {e}")
        import traceback
        traceback.print_exc()
    wait_for_enter()

# =========================
# Main menu
# =========================
def main() -> None:
    while True:
        clear_screen()
        print("==================================================")
        print("  HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH (DSS) QUẬN 7")
        print("                 Console CLI")
        print("==================================================")
        print("\n--- MENU CHÍNH ---")
        print(" [1] Tổng quan Dữ liệu")
        print(" [2] Quản lý Trọng số (AHP)")
        print(" [3] Chạy Xếp hạng (TOPSIS)")
        print(" [4] Chạy Phân tích (What-if)")
        print(" [5] Xuất Báo cáo PDF (Test)")
        print("\n [0] Thoát")

        choice = input("\nNhập lựa chọn: ").strip()
        if choice == "1":
            show_data_overview()
        elif choice == "2":
            manage_ahp()
        elif choice == "3":
            run_topsis_console()
        elif choice == "4":
            run_what_if_console()
        elif choice == "5":
            export_report_console()
        elif choice == "0":
            print("Đang thoát...")
            break
        else:
            print("Lựa chọn không hợp lệ.")
            wait_for_enter()

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    load_weights_file()   # khởi tạo nếu cần
    main()
