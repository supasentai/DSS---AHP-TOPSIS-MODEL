import os
import yaml
# Import hai hàm chính từ hai module bạn vừa tạo
from ahp_module import run_interactive_ahp
from topsis_module import run_topsis_model

# --- CẤU HÌNH ĐƯỜNG DẪN CHUNG ---
BASE_DIR = os.getcwd()  # Lấy thư mục hiện tại
CSV_PATH = os.path.join(BASE_DIR, "AHP_Data_synced_fixed.csv")
JSON_PATH = os.path.join(BASE_DIR, "metadata.json")
CRITERIA_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights.yaml")


def display_main_menu():
    """Hiển thị menu chính và lấy lựa chọn của người dùng."""
    print("\n" + "=" * 50)
    print("--- HỆ THỐNG HỖ TRỢ RA QUYẾT ĐỊNH AHP-TOPSIS ---")
    print("=" * 50)
    print("[1] Chạy phân tích địa điểm (TOPSIS) với mô hình có sẵn")
    print("[2] Tạo hoặc Cập nhật trọng số mô hình (AHP tương tác)")
    print("[0] Thoát chương trình")
    print("=" * 50)
    return input("Vui lòng nhập lựa chọn của bạn: ").strip()


def run_existing_model_analysis():
    """Chức năng chọn và chạy một mô hình TOPSIS đã có."""
    print("\n--- CHẠY PHÂN TÍCH VỚI MÔ HÌNH CÓ SẴN ---")
    if not os.path.exists(CRITERIA_WEIGHTS_PATH):
        print("Lỗi: Không tìm thấy file 'weights.yaml'.")
        print("Vui lòng chạy tùy chọn [2] để tạo ít nhất một mô hình trước.")
        return

    try:
        with open(CRITERIA_WEIGHTS_PATH, 'r', encoding='utf-8') as f:
            all_weights = yaml.safe_load(f)
            if not all_weights:
                print("Lỗi: File 'weights.yaml' rỗng.")
                print("Vui lòng chạy tùy chọn [2] để tạo ít nhất một mô hình.")
                return
    except Exception as e:
        print(f"Lỗi khi đọc file 'weights.yaml': {e}")
        return

    # Hiển thị các mô hình có sẵn
    model_keys = list(all_weights.keys())
    print("Các mô hình hiện có:")
    for i, key in enumerate(model_keys):
        print(f"  [{i + 1}] {key.upper()}")

    # Lấy lựa chọn của người dùng
    while True:
        try:
            choice = input("\nVui lòng chọn mô hình bằng số hoặc tên: ").strip().lower()
            if choice in model_keys:
                selected_model = choice
                break
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(model_keys):
                selected_model = model_keys[choice_index]
                break
            print("Lựa chọn không hợp lệ.")
        except ValueError:
            print("Lựa chọn không hợp lệ.")

    # Chạy mô hình TOPSIS với lựa chọn
    run_topsis_model(CSV_PATH, JSON_PATH, selected_model, all_weights)


if __name__ == "__main__":
    while True:
        choice = display_main_menu()

        if choice == '1':
            run_existing_model_analysis()
        elif choice == '2':
            run_interactive_ahp()
        elif choice == '0':
            print("Cảm ơn đã sử dụng chương trình!")
            break
        else:
            print("Lựa chọn không hợp lệ. Vui lòng nhập 0, 1 hoặc 2.")

        input("\nNhấn Enter để quay lại menu chính...")