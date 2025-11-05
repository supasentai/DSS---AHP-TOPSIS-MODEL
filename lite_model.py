"""
Phiên bản Console (CLI) của ứng dụng DSS Quận 7.
Dùng để chạy logic nghiệp vụ mà không cần giao diện Streamlit.
Bao gồm chức năng test module xuất PDF.
"""

import pandas as pd
import numpy as np
import yaml
import json
import os
import sys

# --- Cấu hình đường dẫn ---
DATA_PATH = "data/AHP_Data_synced_fixed.xlsx"
METADATA_PATH = "data/metadata.json"
# Chỉ dùng 1 file weights
WEIGHTS_PATH = "data/weights.yaml"
# File default sẽ dùng để khởi tạo nếu file weights không tồn tại
DEFAULT_WEIGHTS_PATH = "data/defaultweights.yaml"

# --- Import các module nghiệp vụ ---
try:
    from ahp_module import calculate_ahp_weights, save_weights_to_yaml
    from topsis_module import run_topsis_model
    from sensitivity_module import run_what_if_analysis
    # Import module báo cáo mới để test
    from report_module import create_full_report
except ImportError as e:
    print(f"LỖI: Không thể import module. Vui lòng đảm bảo các file .py")
    print(f"(.py, topsis_module.py, sensitivity_module.py, report_module.py)")
    print(f"nằm cùng thư mục. Lỗi chi tiết: {e}")
    sys.exit(1)

# --- Cấu hình hiển thị Pandas ---
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# ===========================
# Helpers cho Console
# ===========================

def clear_screen():
    """Xóa màn hình console."""
    os.system('cls' if os.name == 'nt' else 'clear')


def wait_for_enter():
    """Chờ người dùng nhấn Enter."""
    input("\n... Nhấn Enter để tiếp tục ...")


def nice_name(col: str) -> str:
    """Làm đẹp tên cột (sao chép từ app.py)"""
    return str(col).replace("_", " ").strip().title()


def get_all_criteria():
    """Tải danh sách tất cả tiêu chí từ file data."""
    try:
        df = pd.read_excel(DATA_PATH)
        return [col for col in df.columns if col not in ['ward', 'ward_id']]
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file dữ liệu: {DATA_PATH}")
        return []
    except Exception as e:
        print(f"LỖI khi đọc file dữ liệu: {e}")
        return []


# --- LOGIC MỚI: TẢI WEIGHTS (chỉ dùng 1 file) ---
def load_weights_file():
    """
    Tải file weights.yaml.
    Nếu không tồn tại, khởi tạo nó từ defaultweights.yaml.
    """
    if not os.path.exists(WEIGHTS_PATH):
        if os.path.exists(DEFAULT_WEIGHTS_PATH):
            print(f"File {WEIGHTS_PATH} không tìm thấy. Đang khởi tạo từ {DEFAULT_WEIGHTS_PATH}...")
            try:
                with open(DEFAULT_WEIGHTS_PATH, 'r', encoding='utf-8') as f_default:
                    default_data = yaml.safe_load(f_default)

                with open(WEIGHTS_PATH, 'w', encoding='utf-8') as f_weights:
                    yaml.dump(default_data, f_weights, allow_unicode=True, sort_keys=False, indent=2)
                return default_data or {}
            except Exception as e:
                print(f"Lỗi khi khởi tạo {WEIGHTS_PATH}: {e}")
                return {}
        else:
            print(f"Cảnh báo: Không tìm thấy {WEIGHTS_PATH} và {DEFAULT_WEIGHTS_PATH}. Bắt đầu với dict rỗng.")
            return {}

    # Nếu file weights.yaml tồn tại, đọc nó
    try:
        with open(WEIGHTS_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"LỖI khi đọc {WEIGHTS_PATH}: {e}")
        return {}


# --- KẾT THÚC LOGIC MỚI ---


def select_model(all_weights):
    """
    Hiển thị danh sách các mô hình và cho người dùng chọn.
    Trả về (tên mô hình, dict trọng số).
    """
    if not all_weights:
        print("Chưa có mô hình AHP nào được lưu.")
        return None, None

    models = list(all_weights.keys())
    print("--- Chọn một mô hình ---")
    for i, name in enumerate(models):
        print(f" [{i + 1}] {name}")
    print(" [0] Quay lại")

    while True:
        try:
            choice = int(input("Nhập lựa chọn của bạn: "))
            if 0 < choice <= len(models):
                model_name = models[choice - 1]
                return model_name, all_weights[model_name]
            elif choice == 0:
                return None, None
            else:
                print("Lựa chọn không hợp lệ.")
        except ValueError:
            print("Vui lòng nhập một con số.")


# ===========================
# Chức năng (Tương tự các Trang)
# ===========================

def show_data_overview():
    """Chức năng 1: Tổng quan Dữ liệu"""
    clear_screen()
    print("===================================")
    print("  1. TỔNG QUAN DỮ LIỆU")
    print("===================================")
    try:
        df = pd.read_excel(DATA_PATH)
        print("\n--- Thống kê Mô tả (5 hàng đầu) ---")
        print(df.describe().T.head())

        print(f"\n--- Dữ liệu thô (5 hàng đầu) ---")
        print(df.head())

        print(f"\nTổng cộng: {len(df)} phường và {len(get_all_criteria())} tiêu chí.")

    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file dữ liệu: {DATA_PATH}")
    except Exception as e:
        print(f"LỖI khi đọc file: {e}")
    wait_for_enter()


def manage_ahp():
    """Chức năng 2: Quản lý Trọng số (AHP)"""
    clear_screen()
    print("===================================")
    print("  2. QUẢN LÝ TRỌNG SỐ (AHP)")
    print("===================================")

    all_weights = load_weights_file()
    print("Các mô hình hiện có:", list(all_weights.keys()) or "[Trống]")
    print("\n [1] Tạo/Tùy chỉnh (Phương pháp 1: Gán điểm 1-10)")
    print(" [2] Tạo/Tùy chỉnh (Phương pháp 2: Ma trận so sánh cặp)")
    print(" [0] Quay lại Menu")

    choice = input("Nhập lựa chọn: ")

    if choice == '1':
        ahp_method_1(all_weights)
    elif choice == '2':
        ahp_method_2(all_weights)
    else:
        return


def ahp_method_1(all_weights):
    """AHP - Phương pháp 1: Gán điểm 1-10"""
    print("\n--- AHP: Tạo mới (Phương pháp 1: Điểm 1-10) ---")
    model_name = input("Nhập tên mô hình (nếu trùng sẽ tự động đổi tên): ").strip()
    if not model_name:
        print("Tên mô hình không được để trống.")
        wait_for_enter()
        return

    # Logic kiểm tra tên trùng sẽ do save_weights_to_yaml xử lý

    all_criteria = get_all_criteria()
    if not all_criteria:
        return

    # Lấy trọng số cũ (nếu có) để làm giá trị mặc định
    original_weights_dict = all_weights.get(model_name, {})
    default_scores = {}
    if original_weights_dict:
        max_weight = max(original_weights_dict.values()) if original_weights_dict else 1
        if max_weight == 0: max_weight = 1
        default_scores = {k: int(round((v / max_weight) * 9 + 1)) for k, v in original_weights_dict.items()}

    print("\nChọn tiêu chí (nhập số, cách nhau bằng dấu phẩy. Ví dụ: 1,3,5):")
    default_selection_keys = list(original_weights_dict.keys()) if original_weights_dict else all_criteria
    for i, name in enumerate(all_criteria):
        is_default = " (*)" if name in default_selection_keys else ""
        print(f" [{i + 1}] {nice_name(name)}{is_default}")

    try:
        selected_indices_str = input(f"Chọn (mặc định: {'tất cả' if not original_weights_dict else 'theo mô hình'}): ")
        if not selected_indices_str.strip():
            selected_criteria = default_selection_keys
        else:
            selected_indices = [int(x.strip()) - 1 for x in selected_indices_str.split(',')]
            selected_criteria = [all_criteria[i] for i in selected_indices if 0 <= i < len(all_criteria)]
    except ValueError:
        print("Lỗi: Nhập không hợp lệ. Phải là các con số.")
        wait_for_enter()
        return

    if not selected_criteria:
        print("Chưa chọn tiêu chí.")
        wait_for_enter()
        return

    print("\nNhập điểm từ 1 (Không quan trọng) đến 10 (Rất quan trọng):")
    scores = {}
    for crit in selected_criteria:
        while True:
            try:
                default_val = default_scores.get(crit, 5)
                score_str = input(f"  Điểm cho '{nice_name(crit)}' (mặc định: {default_val}): ")
                if not score_str.strip():
                    score = default_val
                else:
                    score = float(score_str)

                if 1 <= score <= 10:
                    scores[crit] = score
                    break
                else:
                    print("Điểm phải từ 1 đến 10.")
            except ValueError:
                print("Vui lòng nhập một con số.")

    total_score = sum(scores.values())
    if total_score == 0:
        print("Lỗi: Tổng điểm bằng 0.")
        wait_for_enter()
        return

    normalized_weights = {k: v / total_score for k, v in scores.items()}

    print("\n--- Trọng số đã chuẩn hóa ---")
    print(json.dumps(normalized_weights, indent=2))

    # --- SỬA LỖI LOGIC LƯU ---
    saved_ok, final_name = save_weights_to_yaml(normalized_weights, model_name, WEIGHTS_PATH)
    if saved_ok:
        if final_name != model_name:
            print(f"\nTHÀNH CÔNG: Tên '{model_name}' đã tồn tại. Đã lưu với tên mới: '{final_name}'")
        else:
            print(f"\nTHÀNH CÔNG: Đã lưu mô hình '{final_name}' vào {WEIGHTS_PATH}.")
    else:
        print(f"\nLỖI: Không thể lưu mô hình.")
    # --- KẾT THÚC SỬA LỖI ---
    wait_for_enter()


def ahp_method_2(all_weights):
    """AHP - Phương pháp 2: Ma trận so sánh cặp"""
    print("\n--- AHP: Tạo mới (Phương pháp 2: Ma trận) ---")
    model_name = input("Nhập tên mô hình (nếu trùng sẽ tự động đổi tên): ").strip()
    if not model_name:
        print("Tên mô hình không được để trống.")
        wait_for_enter()
        return

    # Logic kiểm tra tên trùng sẽ do save_weights_to_yaml xử lý

    all_criteria = get_all_criteria()
    if not all_criteria:
        return

    original_weights_dict = all_weights.get(model_name, {})
    default_selection_keys = list(original_weights_dict.keys()) if original_weights_dict else all_criteria

    print("\nChọn tiêu chí (nhập số, cách nhau bằng dấu phẩy. Ví dụ: 1,3,5):")
    for i, name in enumerate(all_criteria):
        is_default = " (*)" if name in default_selection_keys else ""
        print(f" [{i + 1}] {nice_name(name)}{is_default}")

    try:
        selected_indices_str = input(f"Chọn (mặc định: {'tất cả' if not original_weights_dict else 'theo mô hình'}): ")
        if not selected_indices_str.strip():
            selected_criteria = default_selection_keys
        else:
            selected_indices = [int(x.strip()) - 1 for x in selected_indices_str.split(',')]
            selected_criteria = [all_criteria[i] for i in selected_indices if 0 <= i < len(all_criteria)]
    except ValueError:
        print("Lỗi: Nhập không hợp lệ. Phải là các con số.")
        wait_for_enter()
        return

    n = len(selected_criteria)
    if n < 2:
        print("Cần ít nhất 2 tiêu chí để so sánh cặp.")
        wait_for_enter()
        return

    matrix = np.ones((n, n))
    print("\nNhập giá trị so sánh (1-9):")

    for i in range(n):
        for j in range(i + 1, n):
            while True:
                try:
                    val = float(input(
                        f"  '{nice_name(selected_criteria[i])}' quan trọng hơn '{nice_name(selected_criteria[j])}' (1-9): "))
                    if 1 / 9 <= val <= 9:
                        matrix[i, j] = val
                        matrix[j, i] = 1.0 / val
                        break
                    else:
                        print("Giá trị phải từ 1/9 đến 9.")
                except ValueError:
                    print("Vui lòng nhập một con số.")

    weights, cr = calculate_ahp_weights(matrix)
    if weights is None:
        print("\nLỖI: Không thể tính toán AHP (ví dụ: ma trận không hợp lệ).")
        wait_for_enter()
        return

    print(f"\n--- Tính toán hoàn tất ---")
    print(f"  Consistency Ratio (CR): {cr:.4f}")
    if cr >= 0.1:
        print("  CẢNH BÁO: CR >= 0.1. Kết quả không nhất quán. Hãy xem xét làm lại.")
    else:
        print("  CR < 0.1. Kết quả nhất quán. Tốt.")

    weights_dict = {name: weight for name, weight in zip(selected_criteria, weights)}
    print("\n--- Trọng số ---")
    print(json.dumps(weights_dict, indent=2))

    # --- SỬA LỖI LOGIC LƯU ---
    saved_ok, final_name = save_weights_to_yaml(weights_dict, model_name, WEIGHTS_PATH)
    if saved_ok:
        if final_name != model_name:
            print(f"\nTHÀNH CÔNG: Tên '{model_name}' đã tồn tại. Đã lưu với tên mới: '{final_name}'")
        else:
            print(f"\nTHÀNH CÔNG: Đã lưu mô hình '{final_name}' vào {WEIGHTS_PATH}.")
    else:
        print(f"\nLỖI: Không thể lưu mô hình.")
    # --- KẾT THÚC SỬA LỖI ---
    wait_for_enter()


def run_topsis_console():
    """Chức năng 3: Chạy Xếp hạng (TOPSIS)"""
    clear_screen()
    print("===================================")
    print("  3. CHẠY XẾP HẠNG (TOPSIS)")
    print("===================================")

    all_weights = load_weights_file()
    model_name, weights_dict = select_model(all_weights)

    if model_name:
        print(f"\nĐang chạy TOPSIS cho mô hình: {model_name}...")
        report_df = run_topsis_model(
            data_path=DATA_PATH,
            json_path=METADATA_PATH,
            analysis_type=model_name,
            all_criteria_weights=all_weights
        )

        if report_df is not None:
            print("\n--- KẾT QUẢ XẾP HẠNG TOPSIS ---")
            print(report_df)
            print(f"\nĐã lưu kết quả vào: data/ranking_result_{model_name}.xlsx")
        else:
            print("LỖI: Chạy TOPSIS thất bại.")

    wait_for_enter()


def run_what_if_console():
    """Chức năng 4: Chạy Phân tích (What-if)"""
    clear_screen()
    print("===================================")
    print("  4. CHẠY PHÂN TÍCH (WHAT-IF)")
    print("===================================")

    all_weights = load_weights_file()
    print("Chọn mô hình GỐC để chạy What-if:")
    model_name, original_weights = select_model(all_weights)

    if not model_name:
        return

    print(f"\n--- Điều chỉnh trọng số cho What-if (dựa trên '{model_name}') ---")
    print("Nhập trọng số MỚI. Nhấn Enter để giữ nguyên giá trị cũ.")

    new_weights_dict = {}
    for crit, old_weight in original_weights.items():
        while True:
            try:
                new_weight_str = input(f"  '{nice_name(crit)}' (cũ: {old_weight:.4f}): ").strip()
                if not new_weight_str:
                    new_weights_dict[crit] = old_weight
                    break
                else:
                    new_weights_dict[crit] = float(new_weight_str)
                    break
            except ValueError:
                print("Vui lòng nhập một con số.")

    # Chuẩn hóa trọng số mới
    total_new_weight = sum(new_weights_dict.values())
    normalized_weights = {k: (v / total_new_weight if total_new_weight > 0 else 0)
                          for k, v in new_weights_dict.items()}

    print("\nĐang chạy phân tích What-if...")
    # CẬP NHẬT: Truyền all_weights vào hàm what-if
    original_df, new_df = run_what_if_analysis(model_name, normalized_weights, all_weights)

    if original_df is not None and new_df is not None:
        print("\n--- BẢNG XẾP HẠNG GỐC ---")
        print(original_df)
        print("\n--- BẢNG XẾP HẠNG MỚI (WHAT-IF) ---")
        print(new_df)

        # So sánh
        df_orig_simple = original_df[['Tên phường', 'Rank']].rename(columns={'Rank': 'Hạng Gốc'})
        df_new_simple = new_df[['Tên phường', 'Rank']].rename(columns={'Rank': 'Hạng Mới'})
        df_rank_change = pd.merge(df_orig_simple, df_new_simple, on='Tên phường')
        df_rank_change['Thay đổi'] = df_rank_change['Hạng Gốc'] - df_rank_change['Hạng Mới']
        print("\n--- SO SÁNH THAY ĐỔI (Gốc - Mới) ---")
        print(df_rank_change.sort_values(by='Hạng Mới'))

    else:
        print("LỖI: Chạy What-if thất bại.")

    wait_for_enter()


def export_report_console():
    """Chức năng 5: Xuất Báo cáo PDF (TEST MODULE)"""
    clear_screen()
    print("===================================")
    print("  5. XUẤT BÁO CÁO PDF (Test)")
    print("===================================")
    print("LƯU Ý: Chức năng này yêu cầu mô hình AHP và")
    print("file kết quả TOPSIS (.xlsx) tương ứng phải tồn tại.")

    all_weights = load_weights_file()
    model_name, _ = select_model(all_weights)

    if not model_name:
        return

    # Kiểm tra xem file kết quả có tồn tại không
    ranking_file = f"data/ranking_result_{model_name}.xlsx"
    if not os.path.exists(ranking_file):
        print(f"\nLỖI: Không tìm thấy file '{ranking_file}'.")
        print(f"Vui lòng chạy TOPSIS (Lựa chọn 3) cho mô hình '{model_name}' trước.")
        wait_for_enter()
        return

    print(f"\nĐang tạo báo cáo PDF cho mô hình: {model_name}...")
    try:
        pdf_path = create_full_report(model_name, all_weights)
        if pdf_path:
            print(f"\n--- THÀNH CÔNG! ---")
            print(f"Đã lưu báo cáo tại: {pdf_path}")
        else:
            print("\nLỖI: Không thể tạo báo cáo PDF. (Xem chi tiết lỗi ở trên).")

    except Exception as e:
        print(f"\nLỖI NGOẠI LỆ khi tạo báo cáo: {e}")
        import traceback
        traceback.print_exc()

    wait_for_enter()


# ===========================
# MENU CHÍNH
# ===========================

def main():
    """Vòng lặp menu chính của ứng dụng console."""
    while True:
        clear_screen()
        print("==================================================")
        print("  HỆ THỐNG HỖ TRỢ QUYẾT ĐỊNH (DSS) QUẬN 7")
        print("             (Phiên bản Console)")
        print("==================================================")
        print("\n--- MENU CHÍNH ---")
        print(" [1] Tổng quan Dữ liệu")
        print(" [2] Quản lý Trọng số (AHP)")
        print(" [3] Chạy Xếp hạng (TOPSIS)")
        print(" [4] Chạy Phân tích (What-if)")
        print(" [5] Xuất Báo cáo PDF (Test)")
        print("\n [0] Thoát")

        choice = input("\nNhập lựa chọn của bạn: ").strip()

        if choice == '1':
            show_data_overview()
        elif choice == '2':
            manage_ahp()
        elif choice == '3':
            run_topsis_console()
        elif choice == '4':
            run_what_if_console()
        elif choice == '5':
            export_report_console()
        elif choice == '0':
            print("Đang thoát... Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
            wait_for_enter()


if __name__ == "__main__":
    # Đảm bảo các thư mục cần thiết tồn tại
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Chạy logic khởi tạo file weights một lần khi bắt đầu
    load_weights_file()

    main()