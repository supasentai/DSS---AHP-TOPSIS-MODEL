# lite_model.py
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
# ### SỬA ĐỔI: Phân biệt default và user weights ###
DEFAULT_WEIGHTS_PATH = "data/defaultweights.yaml"
USER_WEIGHTS_PATH = "data/weights.yaml"
WEIGHTS_PATH = USER_WEIGHTS_PATH  # Giữ tương thích cho hàm save (mặc dù hàm save có default)

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


def load_weights_file():
    """
    Tải file default và user.
    Trả về (all_weights, default_model_names)
    all_weights: Dict đã gộp (user ghi đè default)
    default_model_names: Set các tên mô hình từ file default
    """
    default_weights = {}
    user_weights = {}

    # 1. Tải default
    if os.path.exists(DEFAULT_WEIGHTS_PATH):
        try:
            with open(DEFAULT_WEIGHTS_PATH, 'r', encoding='utf-8') as f:
                default_weights = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"LỖI khi đọc {DEFAULT_WEIGHTS_PATH}: {e}")
    else:
        print(f"Cảnh báo: Không tìm thấy file {DEFAULT_WEIGHTS_PATH}.")

    # 2. Tải user
    if os.path.exists(USER_WEIGHTS_PATH):
        try:
            with open(USER_WEIGHTS_PATH, 'r', encoding='utf-8') as f:
                user_weights = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"LỖI khi đọc {USER_WEIGHTS_PATH}: {e}")
    else:
        print(f"Thông báo: Không tìm thấy file {USER_WEIGHTS_PATH}. Chỉ sử dụng default (nếu có).")

    # 3. Lấy tên default và gộp
    default_model_names = set(default_weights.keys())
    all_weights = {**default_weights, **user_weights}  # User ghi đè default

    return all_weights, default_model_names


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

    # ### SỬA ĐỔI: Lấy cả all_weights và default_model_names ###
    all_weights, default_model_names = load_weights_file()

    print("Các mô hình hiện có:", list(all_weights.keys()) or "[Trống]")
    print("\n [1] Tạo/Cập nhật mô hình (Phương pháp 1: Gán điểm 1-10)")
    print(" [2] Tạo/Cập nhật mô hình (Phương pháp 2: Ma trận so sánh cặp)")
    print(" [0] Quay lại Menu")

    choice = input("Nhập lựa chọn: ")

    if choice == '1':
        # ### SỬA ĐỔI: Truyền default_model_names ###
        ahp_method_1(all_weights, default_model_names)
    elif choice == '2':
        # ### SỬA ĐỔI: Truyền default_model_names ###
        ahp_method_2(all_weights, default_model_names)
    else:
        return


def ahp_method_1(all_weights, default_model_names):
    """AHP - Phương pháp 1: Gán điểm 1-10"""
    print("\n--- AHP: Tạo/Cập nhật (Phương pháp 1: Điểm 1-10) ---")
    model_name = input("Nhập tên mô hình: ").strip()
    if not model_name:
        print("Tên mô hình không được để trống.")
        wait_for_enter()
        return

    # ### SỬA ĐỔI: Logic kiểm tra tên ###
    if model_name in default_model_names:
        print(f"LỖI: '{model_name}' là một mô hình default (gốc).")
        print("Bạn không thể ghi đè lên mô hình default.")
        print("Nếu muốn tùy chỉnh, vui lòng chạy lại và nhập một tên mới.")
        wait_for_enter()
        return
    # ### HẾT SỬA ĐỔI ###

    if model_name in all_weights:
        # Tên này nằm trong all_weights nhưng không nằm trong default_model_names
        # => Nó là mô hình của user (trong weights.yaml)
        print(f"Cảnh báo: Tên '{model_name}' đã tồn tại (trong weights.yaml). Sẽ ghi đè.")
    else:
        print(f"Thông tin: Sẽ tạo mô hình mới tên '{model_name}'.")

    all_criteria = get_all_criteria()
    if not all_criteria:
        return

    print("\nChọn tiêu chí (nhập số, cách nhau bằng dấu phẩy. Ví dụ: 1,3,5):")
    for i, name in enumerate(all_criteria):
        print(f" [{i + 1}] {nice_name(name)}")

    try:
        selected_indices = [int(x.strip()) - 1 for x in input("Chọn: ").split(',')]
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
                score = float(input(f"  Điểm cho '{nice_name(crit)}': "))
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

    # ### SỬA ĐỔI: Luôn lưu vào USER_WEIGHTS_PATH ###
    if save_weights_to_yaml(normalized_weights, model_name, USER_WEIGHTS_PATH):
        print(f"\nTHÀNH CÔNG: Đã lưu mô hình '{model_name}' vào {USER_WEIGHTS_PATH}.")
    else:
        print(f"\nLỖI: Không thể lưu mô hình.")
    wait_for_enter()


def ahp_method_2(all_weights, default_model_names):
    """AHP - Phương pháp 2: Ma trận so sánh cặp"""
    print("\n--- AHP: Tạo/Cập nhật (Phương pháp 2: Ma trận) ---")
    model_name = input("Nhập tên mô hình: ").strip()
    if not model_name:
        print("Tên mô hình không được để trống.")
        wait_for_enter()
        return

    # ### SỬA ĐỔI: Logic kiểm tra tên ###
    if model_name in default_model_names:
        print(f"LỖI: '{model_name}' là một mô hình default (gốc).")
        print("Bạn không thể ghi đè lên mô hình default.")
        print("Nếu muốn tùy chỉnh, vui lòng chạy lại và nhập một tên mới.")
        wait_for_enter()
        return
    # ### HẾT SỬA ĐỔI ###

    if model_name in all_weights:
        # Tên này nằm trong all_weights nhưng không nằm trong default_model_names
        # => Nó là mô hình của user (trong weights.yaml)
        print(f"Cảnh báo: Tên '{model_name}' đã tồn tại (trong weights.yaml). Sẽ ghi đè.")
    else:
        print(f"Thông tin: Sẽ tạo mô hình mới tên '{model_name}'.")

    all_criteria = get_all_criteria()
    if not all_criteria:
        return

    print("\nChọn tiêu chí (nhập số, cách nhau bằng dấu phẩy. Ví dụ: 1,3,5):")
    for i, name in enumerate(all_criteria):
        print(f" [{i + 1}] {nice_name(name)}")

    try:
        selected_indices = [int(x.strip()) - 1 for x in input("Chọn: ").split(',')]
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

    # ### SỬA ĐỔI: Luôn lưu vào USER_WEIGHTS_PATH ###
    if save_weights_to_yaml(weights_dict, model_name, USER_WEIGHTS_PATH):
        print(f"\nTHÀNH CÔNG: Đã lưu mô hình '{model_name}' vào {USER_WEIGHTS_PATH}.")
    else:
        print(f"\nLỖI: Không thể lưu mô hình.")
    wait_for_enter()


def run_topsis_console():
    """Chức năng 3: Chạy Xếp hạng (TOPSIS)"""
    clear_screen()
    print("===================================")
    print("  3. CHẠY XẾP HẠNG (TOPSIS)")
    print("===================================")

    # ### SỬA ĐỔI: Chỉ cần all_weights, bỏ qua default_names ###
    all_weights, _ = load_weights_file()
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

    # ### SỬA ĐỔI: Chỉ cần all_weights, bỏ qua default_names ###
    all_weights, _ = load_weights_file()
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
    original_df, new_df = run_what_if_analysis(model_name, normalized_weights)

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

    # ### SỬA ĐỔI: Chỉ cần all_weights, bỏ qua default_names ###
    all_weights, _ = load_weights_file()
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
        pdf_path = create_full_report(model_name)
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

    # ### THÊM MỚI: Logic khởi tạo user weights từ default weights ###
    if not os.path.exists(USER_WEIGHTS_PATH) and os.path.exists(DEFAULT_WEIGHTS_PATH):
        try:
            import shutil

            shutil.copyfile(DEFAULT_WEIGHTS_PATH, USER_WEIGHTS_PATH)
            print(f"Thông báo: Đã khởi tạo {USER_WEIGHTS_PATH} từ file default.")
        except Exception as e:
            print(f"Lỗi khi copy file default weights: {e}")

    main()


