# app.py (ĐÃ SỬA LỖI STATE)
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os

# Import các hàm bạn đã tạo
# Đảm bảo file 'topsis.py' đã được sửa để return dataframe
try:
    from ahp_module import calculate_ahp_weights, save_weights_to_yaml
    from topsis_module import run_topsis_model
except ImportError as e:
    st.error(f"Lỗi import module: {e}. Vui lòng đảm bảo các file `ahp_module.py` và `topsis.py` nằm cùng thư mục.")
    st.stop()

# --- Cấu hình trang ---
st.set_page_config(
    page_title="DSS",
    page_icon="🦈",
    layout="wide"
)

# --- KHỞI TẠO SESSION STATE ---
if 'criteria_names' not in st.session_state:
    st.session_state.criteria_names = []
if 'ahp_matrix' not in st.session_state:
    st.session_state.ahp_matrix = None  # Sẽ được khởi tạo sau khi tải tiêu chí

# ====================================================================
# --- GIAO DIỆN NGƯỜI DÙNG (UI) ---
# ====================================================================

st.title("🦈 Chọn địa điểm quận 7")

# --- Sidebar để điều hướng ---
st.sidebar.title("Menu")
page = st.sidebar.radio("Chọn một trang:", ["Phân tích Địa điểm (TOPSIS)", "Tùy chỉnh Trọng số (AHP)"])

# ====================================================================
# --- TRANG 1: TÙY CHỈNH TRỌNG SỐ (AHP) --- (ĐÃ SỬA LỖI)
# ====================================================================
if page == "Tùy chỉnh Trọng số (AHP)":
    st.header("Tạo và Cập nhật Trọng số Mô hình bằng AHP")

    # Tải tên tiêu chí từ file CSV
    try:
        df_data = pd.read_csv("AHP_Data_synced_fixed.csv", encoding='latin1')
        st.session_state.criteria_names = [col for col in df_data.columns if col not in ['ward', 'ward_id']]
        n = len(st.session_state.criteria_names)
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file 'AHP_Data_synced_fixed.csv'.")
        st.stop()

    # --- SỬA LỖI QUAN TRỌNG: KHỞI TẠO MA TRẬN TRONG SESSION STATE ---
    # Chỉ tạo ma trận mới nếu nó chưa tồn tại hoặc kích thước thay đổi
    if st.session_state.ahp_matrix is None or st.session_state.ahp_matrix.shape[0] != n:
        st.session_state.ahp_matrix = np.ones((n, n))

    model_name = st.text_input("Nhập tên mô hình để tạo/cập nhật:", placeholder="Ví dụ: office, retail_store")

    st.subheader("Nhập ma trận so sánh cặp:")

    # Tạo hàng tiêu đề cho các cột
    column_specs_header = [1.5] + [1] * n
    header_cols = st.columns(column_specs_header)
    for j, col_name in enumerate(st.session_state.criteria_names):
        with header_cols[j + 1]:
            st.write(f"**{col_name}**")

    # --- SỬA LỖI: VÒNG LẶP CẬP NHẬT TRẠNG THÁI ---
    # Vòng lặp này sẽ đọc giá trị từ widget và cập nhật ma trận trong session_state

    # Bước 1: Thu thập tất cả các giá trị người dùng nhập ở nửa trên
    for i in range(n):
        for j in range(i + 1, n):
            key = f"matrix_{i}_{j}"
            value = st.session_state.get(key, 1.0)  # Lấy giá trị từ state nếu có

            # Cập nhật ma trận trong state ngay lập tức
            st.session_state.ahp_matrix[i, j] = value
            if value != 0:
                st.session_state.ahp_matrix[j, i] = 1.0 / value

    # Bước 2: Hiển thị toàn bộ ma trận (với các giá trị đã được tính toán)
    for i in range(n):
        column_specs_row = [1.5] + [1] * n
        row_cols = st.columns(column_specs_row)

        with row_cols[0]:
            st.write("")
            st.write(f"**{st.session_state.criteria_names[i]}**")

        for j in range(n):
            with row_cols[j + 1]:
                key = f"cell_{i}_{j}"

                if i == j:
                    # Đường chéo
                    st.text_input(key, value="1.00", disabled=True, label_visibility="collapsed")
                elif i < j:
                    # Nửa trên: ô cho phép nhập
                    st.number_input(
                        label=f"Input {i}-{j}",
                        min_value=0.01,
                        value=st.session_state.ahp_matrix[i, j],  # Giá trị lấy từ state
                        step=0.1,
                        format="%.2f",
                        label_visibility="collapsed",
                        key=f"matrix_{i}_{j}"  # Key này phải khớp với key ở Bước 1
                    )
                else:
                    # Nửa dưới: ô hiển thị (đã bị vô hiệu hóa)
                    st.text_input(
                        key,
                        value=f"{st.session_state.ahp_matrix[i, j]:.2f}",  # Giá trị lấy từ state
                        disabled=True,
                        label_visibility="collapsed"
                    )

    # Nút tính toán
    if st.button("Tính toán và Lưu Trọng số"):
        if not model_name:
            st.warning("Vui lòng nhập tên mô hình.")
        else:
            # Lấy ma trận cuối cùng từ session_state để tính toán
            final_matrix = st.session_state.ahp_matrix
            weights, cr = calculate_ahp_weights(final_matrix)

            if weights is not None and cr is not None and cr < 0.1:
                st.success(f"Kiểm tra nhất quán: TỐT (CR = {cr:.4f})")
                save_weights_to_yaml(weights, st.session_state.criteria_names, model_name)
                st.success(f"Đã lưu thành công trọng số cho mô hình '{model_name}'.")
                st.balloons()
            else:
                cr_val = cr if cr is not None else "N/A"
                st.error(f"CẢNH BÁO: Tỷ số nhất quán (CR = {cr_val:.4f}) không đạt yêu cầu (>= 0.1).")

# ====================================================================
# --- TRANG 2: PHÂN TÍCH ĐỊA ĐIỂM (TOPSIS) ---
# ====================================================================
elif page == "Phân tích Địa điểm (TOPSIS)":
    st.header("Xếp hạng Địa điểm Tối ưu bằng TOPSIS")

    # Tải các mô hình có sẵn
    try:
        with open("weights.yaml", 'r', encoding='utf-8') as f:
            all_weights = yaml.safe_load(f)
            if not all_weights:
                st.warning("Chưa có mô hình nào. Vui lòng qua trang 'Tùy chỉnh Trọng số (AHP)' để tạo.")
                st.stop()
            model_names = list(all_weights.keys())
    except FileNotFoundError:
        st.error("Không tìm thấy file 'weights.yaml'. Vui lòng tạo một mô hình AHP trước.")
        st.stop()

    selected_model = st.selectbox("Chọn một mô hình có sẵn để phân tích:", model_names)

    if st.button(f"Chạy Phân tích cho mô hình '{selected_model.upper()}'"):
        with st.spinner("Đang tính toán, vui lòng chờ..."):

            # --- TÍCH HỢP HÀM TOPSIS THẬT ---
            report_df = run_topsis_model(
                csv_path="AHP_Data_synced_fixed.csv",
                json_path="metadata.json",  # Đảm bảo bạn có file này
                analysis_type=selected_model,
                all_criteria_weights=all_weights
            )

            if report_df is not None:
                st.success("Phân tích hoàn tất!")
                st.subheader("Kết quả Xếp hạng Địa điểm")
                st.dataframe(report_df)
                st.info(
                    f"**Kết luận:** Dựa trên mô hình **{selected_model.upper()}**, địa điểm tối ưu nhất là **{report_df.iloc[0]['Tên phường']}**.")
            else:
                st.error("Đã xảy ra lỗi trong quá trình phân tích TOPSIS. Vui lòng kiểm tra file 'metadata.json'.")