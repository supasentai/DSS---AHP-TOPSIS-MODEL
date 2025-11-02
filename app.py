# app.py (Đã cập nhật Map View Colors)
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import json
import altair as alt
import pydeck as pdk  # <-- IMPORT THƯ VIỆN MỚI

# --- Import các module chức năng ---
try:
    from ahp_module import calculate_ahp_weights, save_weights_to_yaml
    from topsis_module import run_topsis_model
    from sensitivity_module import run_what_if_analysis
except ImportError as e:
    st.error(
        f"Lỗi import module: {e}. Vui lòng đảm bảo các fil `ahp_module.py`, `topsis_module.py`, và `sensitivity_module.py` nằm cùng thư mục.")
    st.stop()

# --- Cấu hình trang ---
st.set_page_config(
    page_title="DSS Quận 7",
    page_icon="🦈",
    layout="wide"
)

# --- KHỞI TẠO SESSION STATE ---
if 'criteria_names' not in st.session_state:
    st.session_state.criteria_names = []
if 'ahp_matrices' not in st.session_state:
    st.session_state.ahp_matrices = {}
if 'customize_mode' not in st.session_state:
    st.session_state.customize_mode = False
if 'selected_model_for_topsis' not in st.session_state:
    st.session_state.selected_model_for_topsis = None
if 'auto_run_topsis' not in st.session_state:
    st.session_state.auto_run_topsis = False
if 'last_saved_model' not in st.session_state:
    st.session_state.last_saved_model = None
if 'last_saved_weights' not in st.session_state:
    st.session_state.last_saved_weights = None
if 'model_for_next_page' not in st.session_state:
    st.session_state.model_for_next_page = None


# ====================================================================
# --- CÁC HÀM CALLBACK CHUYỂN TRANG (ĐỊNH NGHĨA Ở ĐẦU) ---
# ====================================================================

def switch_to_topsis_page_and_run():
    selected_scenario = st.session_state.scenario_selectbox
    st.session_state.selected_model_for_topsis = selected_scenario
    st.session_state.customize_mode = False
    st.session_state.auto_run_topsis = True
    st.session_state.page_navigator = "Phân tích Địa điểm (TOPSIS)"
    st.session_state.last_saved_model = None
    st.session_state.last_saved_weights = None


def switch_to_topsis_with_last_saved():
    model_name = st.session_state.last_saved_model
    if model_name:
        st.session_state.selected_model_for_topsis = model_name
        st.session_state.customize_mode = False
        st.session_state.auto_run_topsis = True
        st.session_state.page_navigator = "Phân tích Địa điểm (TOPSIS)"
        st.session_state.last_saved_model = None
        st.session_state.last_saved_weights = None


def switch_to_map_view():
    st.session_state.model_for_next_page = st.session_state.topsis_model_selector
    st.session_state.page_navigator = "Map View"


def switch_to_sensitivity():
    st.session_state.whatif_model_selector = st.session_state.topsis_model_selector
    st.session_state.page_navigator = "Phân tích Độ nhạy (What-if)"


def switch_to_ahp_customize():
    # Kiểm tra xem selectbox nào đang hoạt động (Trang 4 hay Trang 5)
    if st.session_state.page_navigator == "Phân tích Địa điểm (TOPSIS)":
        st.session_state.scenario_selectbox = st.session_state.topsis_model_selector
    elif st.session_state.page_navigator == "Phân tích Độ nhạy (What-if)":
        st.session_state.scenario_selectbox = st.session_state.whatif_model_selector

    st.session_state.customize_mode = True
    st.session_state.page_navigator = "Tùy chỉnh Trọng số (AHP)"


# ====================================================================
# --- GIAO DIỆN NGƯỜỜI DÙNG (UI) ---
# ====================================================================

st.title("🦈 Hệ thống Hỗ trợ Quyết định Chọn địa điểm Quận 7")

# --- Bố cục Sidebar ---
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Chọn một trang:",
    [
        "Homepage",
        "Tổng quan Dữ liệu",
        "Tùy chỉnh Trọng số (AHP)",
        "Phân tích Địa điểm (TOPSIS)",
        "Phân tích Độ nhạy (What-if)",
        "Map View"
    ],
    key="page_navigator"
)

# --- Logic hiển thị trang ---

# ====================================================================
# --- TRANG 1: HOMEPAGE ---
# ====================================================================
if page == "Homepage":
    st.header("Chào mừng bạn đến với Hệ thống Hỗ trợ Ra quyết định")
    st.markdown("Hãy sử dụng menu bên trái để bắt đầu thực hiện phân tích.")
    st.subheader("📖 Hướng dẫn sử dụng App")
    st.markdown(
        """
        Ứng dụng này giúp bạn ra quyết định chọn địa điểm tối ưu dựa trên phương pháp AHP và TOPSIS. Vui lòng thực hiện theo các bước sau:

        ### 1. Tổng quan Dữ liệu
        - Xem xét dữ liệu gốc và mô tả của các tiêu chí để hiểu rõ bối cảnh.

        ### 2. Tùy chỉnh Trọng số (AHP)
        - Tạo một "mô hình" (ví dụ: `office`, `retail_store`) bằng cách thiết lập mức độ quan trọng (trọng số) cho từng tiêu chí.
        - Bạn có thể chọn chỉ sử dụng một vài tiêu chí bạn quan tâm.
        - Lưu lại mô hình của bạn, sau đó nhấn nút "Chuyển đến Trang Phân tích".

        ### 3. Phân tích Địa điểm (TOPSIS)
        - Chọn mô hình trọng số bạn vừa tạo ở Bước 2.
        - Chạy phân tích để xem bảng xếp hạng.
        - Từ kết quả, bạn có thể chọn "Map View" để xem bản đồ, "Sensitivity" để phân tích độ nhạy, hoặc "Customize" để quay lại chỉnh sửa trọng số.

        ### 4. Phân tích Độ nhạy (What-if)
        - Chọn một mô hình gốc.
        - Sử dụng các thanh trượt để "thử" thay đổi trọng số và xem kết quả xếp hạng thay đổi như thế nào so với bản gốc.

        ### 5. Map View
        - Xem kết quả xếp hạng TOPSIS được trực quan hóa trên bản đồ.
        """
    )

# ====================================================================
# --- TRANG 2: TỔNG QUAN DỮ LIỆU ---
# ====================================================================
elif page == "Tổng quan Dữ liệu":
    st.header("Trang 2: Khám phá và Tổng quan Dữ liệu")

    try:
        # THAY ĐỔI: Đọc file .xlsx
        df = pd.read_excel("AHP_Data_synced_fixed.xlsx")
        with open("metadata.json", 'r', encoding='utf-8-sig') as f:
            metadata = json.load(f)
    except FileNotFoundError as e:
        st.error(
            f"Lỗi: Không tìm thấy file. Vui lòng đảm bảo `AHP_Data_synced_fixed.xlsx` và `metadata.json` nằm trong thư mục.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        st.stop()

    tab1, tab2 = st.tabs(["📊 Thống kê Chung", "📈 Phân tích Từng tiêu chí"])

    with tab1:
        st.subheader("Thông tin Cơ bản")
        col1, col2 = st.columns(2)
        col1.metric("Tổng số Địa điểm (Phường)", df['ward'].nunique())
        col2.metric("Tổng số Tiêu chí", len(df.columns) - 2)
        st.subheader("Thống kê Mô tả các Tiêu chí")
        st.dataframe(df.describe(), use_container_width=True)
        st.subheader("Bảng Dữ liệu gốc (Raw Data)")
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.subheader("Xem xét Chi tiết Từng tiêu chí")
        criteria_list = [col for col in df.columns if col not in ['ward', 'ward_id']]
        selected_criterion = st.selectbox("Chọn một tiêu chí để phân tích:", criteria_list)

        if selected_criterion:
            meta_info = metadata.get(selected_criterion, {})
            full_name = meta_info.get('display_name', selected_criterion.replace('_', ' ').title())
            desc = meta_info.get('description', "Không có mô tả.")
            c_type = meta_info.get('type', 'N/A')

            st.markdown(f"#### {full_name}")

            font_family = "Inter"
            style_info_box = f"font-family: '{font_family}', sans-serif; font-size: 16px; background-color: #E3F2FD; border: 1px solid #90CAF9; border-radius: 0.25rem; padding: 1rem; margin-bottom: 1rem;"
            style_description = f"font-family: '{font_family}', sans-serif; font-size: 16px; line-height: 1.6;"

            desc_col1, desc_col2 = st.columns([1, 3])
            with desc_col1:
                st.markdown(f"""<div style="{style_info_box}"><strong>Loại tiêu chí:</strong> {c_type.title()}</div>""",
                            unsafe_allow_html=True)
            with desc_col2:
                st.markdown(f"""<p style="{style_description}"><strong>Mô tả:</strong> <em>{desc}</em></p>""",
                            unsafe_allow_html=True)
            st.divider()

            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader(f"Top 5 Địa điểm")
                st.markdown(f"({full_name})")
                is_cost = (c_type == 'cost')
                sorted_df = df.sort_values(by=selected_criterion, ascending=is_cost).head(5)
                st.dataframe(sorted_df[['ward', selected_criterion]], use_container_width=True)
            with col2:
                st.subheader("Phân phối Dữ liệu trên các Phường")
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('ward', title="Tên Phường", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y(selected_criterion, title=full_name),
                    tooltip=['ward', selected_criterion]
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

# ====================================================================
# --- TRANG 3: TÙY CHỈNH TRỌNG SỐ (AHP) ---
# ====================================================================
elif page == "Tùy chỉnh Trọng số (AHP)":
    st.header("Trang 3: Tạo và Cập nhật Trọng số Mô hình")

    all_weights = {}
    weights_file = "weights.yaml"
    if os.path.exists(weights_file):
        try:
            with open(weights_file, 'r', encoding='utf-8') as f:
                all_weights = yaml.safe_load(f)
                if not all_weights:
                    all_weights = {}
        except Exception as e:
            st.error(f"Lỗi khi đọc file 'weights.yaml': {e}")
            all_weights = {}

    model_list = ["--- Tạo mô hình mới ---"] + list(all_weights.keys())
    st.subheader("1. Lựa chọn Kịch bản (Scenario)")

    # --- SỬA LỖI F5 (TRANG 3) ---
    selectbox_key_ahp = "scenario_selectbox"
    default_index_ahp = 0

    if 'scenario_selectbox' in st.session_state and st.session_state.scenario_selectbox in model_list:
        default_index_ahp = model_list.index(st.session_state.scenario_selectbox)
    elif selectbox_key_ahp in st.session_state:
        current_saved_scenario = st.session_state[selectbox_key_ahp]
        if current_saved_scenario in model_list:
            default_index_ahp = model_list.index(current_saved_scenario)


    # -----------------------------

    def on_scenario_change():
        st.session_state.selected_model_for_topsis = None
        st.session_state.last_saved_model = None
        st.session_state.last_saved_weights = None


    selected_scenario = st.selectbox(
        "Chọn một kịch bản có sẵn hoặc tạo mới:",
        model_list,
        index=default_index_ahp,
        key=selectbox_key_ahp,
        on_change=on_scenario_change
    )


    def show_customization_tabs(all_weights_passed_in, model_name_placeholder=""):
        if model_name_placeholder:
            st.subheader(f"2. Tùy chỉnh Trọng số cho mô hình: '{model_name_placeholder}'")
            st.session_state.model_name = model_name_placeholder
        else:
            st.subheader("2. Tùy chỉnh Trọng số")
            st.session_state.model_name = st.text_input(
                "Nhập tên cho mô hình mới (ví dụ: 'office_v2', 'retail_store'):")

        if st.session_state.model_name:
            st.divider()
            st.subheader("2.5 Chọn Tiêu chí sử dụng")

            try:
                # THAY ĐỔI: Đọc file .xlsx
                df_data = pd.read_excel("AHP_Data_synced_fixed.xlsx")
                full_criteria_list = [col for col in df_data.columns if col not in ['ward', 'ward_id']]
            except FileNotFoundError:
                st.error("Lỗi: Không tìm thấy file 'AHP_Data_synced_fixed.xlsx'.")
                st.stop()

            default_selection = []
            if model_name_placeholder:
                original_weights_dict = all_weights_passed_in.get(model_name_placeholder, {})
                default_selection = list(original_weights_dict.keys())

            if not default_selection:
                default_selection = full_criteria_list

            st.markdown("Chọn các tiêu chí bạn muốn đưa vào mô hình này:")
            cols = st.columns(3)
            selected_criteria_list = []

            for i, criterion in enumerate(full_criteria_list):
                is_checked_by_default = criterion in default_selection
                with cols[i % 3]:
                    if st.checkbox(
                            criterion,
                            value=is_checked_by_default,
                            key=f"check_{criterion}_{st.session_state.model_name}"
                    ):
                        selected_criteria_list.append(criterion)

            st.divider()

            if not selected_criteria_list:
                st.warning("Vui lòng chọn ít nhất một tiêu chí để bắt đầu thiết lập trọng số.")
                st.stop()

            tab1, tab2 = st.tabs(
                ["Phương pháp 1: Đánh giá trực tiếp (1-10)", "Phương pháp 2: Ma trận so sánh cặp (AHP)"])

            with tab1:
                st.info(
                    "Kéo thanh trượt (1-10) để gán điểm quan trọng cho từng tiêu chí. Các điểm số sẽ được tự động chuẩn hóa thành trọng số.")

                original_weights_dict = all_weights_passed_in.get(st.session_state.model_name, {})

                scores_dict = {}
                if not original_weights_dict:
                    scores_dict = {criterion: 5 for criterion in selected_criteria_list}
                else:
                    max_weight = max(
                        original_weights_dict.values()) if original_weights_dict and original_weights_dict.values() else 1
                    if max_weight == 0: max_weight = 1
                    scores_dict = {k: int(round((v / max_weight) * 9 + 1)) for k, v in original_weights_dict.items()}

                new_scores = {}
                for criterion in selected_criteria_list:
                    score = st.slider(
                        f"Điểm cho '{criterion}'",
                        min_value=1,
                        max_value=10,
                        value=scores_dict.get(criterion, 5),
                        key=f"score_{criterion}_{st.session_state.model_name}"
                    )
                    new_scores[criterion] = score

                total_score = sum(new_scores.values())
                if total_score > 0:
                    normalized_weights = {k: v / total_score for k, v in new_scores.items()}

                    st.subheader("Trọng số (Đã chuẩn hóa)")
                    weights_df_normalized = pd.DataFrame.from_dict(normalized_weights, orient='index',
                                                                   columns=['Trọng số'])
                    st.dataframe(weights_df_normalized, use_container_width=True)

                    if st.button("Lưu Trọng số (Phương pháp 1)", key="save_method_1"):
                        saved_ok = save_weights_to_yaml(normalized_weights, st.session_state.model_name)
                        if saved_ok:
                            st.session_state.last_saved_model = st.session_state.model_name
                            st.session_state.last_saved_weights = normalized_weights
                            st.rerun()
                        else:
                            st.error("Lỗi: Không thể lưu file.")
                else:
                    st.warning("Tổng điểm bằng 0, không thể tính trọng số.")

            with tab2:
                st.info("Nhập ma trận so sánh cặp. Giá trị 1-9 cho biết mức độ quan trọng. (CR < 0.1 để nhất quán)")

                n = len(selected_criteria_list)
                matrix_state_key = f"ahp_matrix_{st.session_state.model_name}_{'_'.join(sorted(selected_criteria_list))}"

                if (matrix_state_key not in st.session_state.ahp_matrices or
                        st.session_state.ahp_matrices[matrix_state_key].shape[0] != n):
                    st.session_state.ahp_matrices[matrix_state_key] = np.ones((n, n))

                current_matrix = st.session_state.ahp_matrices[matrix_state_key]

                column_specs_header = [1.5] + [1] * n
                header_cols = st.columns(column_specs_header)
                for j, col_name in enumerate(selected_criteria_list):
                    with header_cols[j + 1]:
                        st.write(f"**{col_name}**")

                for i in range(n):
                    for j in range(i + 1, n):
                        key = f"matrix_{i}_{j}_{matrix_state_key}"
                        value = st.session_state.get(key, 1.0)
                        current_matrix[i, j] = value
                        if value != 0:
                            current_matrix[j, i] = 1.0 / value

                for i in range(n):
                    column_specs_row = [1.5] + [1] * n
                    row_cols = st.columns(column_specs_row)
                    with row_cols[0]:
                        st.write("")
                        st.write(f"**{selected_criteria_list[i]}**")

                    for j in range(n):
                        with row_cols[j + 1]:
                            key = f"cell_{i}_{j}_{matrix_state_key}"
                            if i == j:
                                st.text_input(key, value="1.00", disabled=True, label_visibility="collapsed")
                            elif i < j:
                                st.number_input(
                                    label=f"Input {i}-{j}",
                                    min_value=0.01,
                                    value=current_matrix[i, j],
                                    step=0.1,
                                    format="%.2f",
                                    label_visibility="collapsed",
                                    key=f"matrix_{i}_{j}_{matrix_state_key}"
                                )
                            else:
                                st.text_input(
                                    key,
                                    value=f"{current_matrix[i, j]:.2f}",
                                    disabled=True,
                                    label_visibility="collapsed"
                                )

                if st.button("Tính toán và Lưu Trọng số (Phương pháp 2)", key="save_method_2"):
                    final_matrix = current_matrix
                    weights, cr = calculate_ahp_weights(final_matrix)

                    if weights is not None and cr is not None and cr < 0.1:
                        st.success(f"Kiểm tra nhất quán: TỐT (CR = {cr:.4f})")
                        weights_dict = {name: weight for name, weight in zip(selected_criteria_list, weights)}

                        saved_ok = save_weights_to_yaml(weights_dict, st.session_state.model_name)
                        if saved_ok:
                            st.session_state.last_saved_model = st.session_state.model_name
                            st.session_state.last_saved_weights = weights_dict
                            st.rerun()
                        else:
                            st.error("Lỗi: Không thể lưu file.")
                    else:
                        cr_val = cr if cr is not None else "N/A"
                        st.error(f"CẢNH BÁO: Tỷ số nhất quán (CR = {cr_val:.4f}) không đạt yêu cầu (>= 0.1).")

            if (st.session_state.get('last_saved_model') == st.session_state.model_name and
                    st.session_state.get('last_saved_weights') is not None):
                st.divider()
                st.success(f"Đã lưu thành công mô hình '{st.session_state.model_name}'!", icon="✅")

                weights_dict = st.session_state.last_saved_weights

                df_chart = pd.DataFrame(weights_dict.items(), columns=["Tiêu chí", "Trọng số"])

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("Bảng Trọng số")
                    df_table = df_chart.copy()
                    # Bỏ cột %, chỉ format số
                    df_table['Trọng số'] = df_table['Trọng số'].map('{:,.4f}'.format)
                    st.dataframe(df_table, use_container_width=True, hide_index=True)

                with col2:
                    st.subheader("Phân bổ Trọng số")

                    base = alt.Chart(df_chart).encode(
                        theta=alt.Theta("Trọng số", stack=True)
                    ).properties(title="Biểu đồ Phân bổ Trọng số")

                    pie = base.mark_arc(outerRadius=120, innerRadius=0).encode(
                        color=alt.Color("Tiêu chí", title="Tiêu chí"),
                        order=alt.Order("Trọng số", sort="descending"),
                        tooltip=["Tiêu chí", alt.Tooltip("Trọng số", format=".1%")]
                    )

                    # --- SỬA LỖI: DÙNG BRACKET NOTATION ['Trọng số'] ---
                    # 1. Hiển thị % BÊN TRONG (màu đen) cho lát > 5%
                    text_inside = base.mark_text(radius=80).encode(
                        text=alt.Text("Trọng số", format=".1%"),
                        order=alt.Order("Trọng số", sort="descending"),
                        color=alt.value("black")  # Đổi thành màu đen
                    ).transform_filter(
                        alt.datum['Trọng số'] > 0.05  # SỬA: Dùng bracket
                    )

                    # 2. Hiển thị % BÊN NGOÀI (màu đen) cho lát <= 5%
                    text_outside = base.mark_text(radius=140).encode(
                        text=alt.Text("Trọng số", format=".1%"),
                        order=alt.Order("Trọng số", sort="descending"),
                        color=alt.value("black")  # Đổi thành màu đen
                    ).transform_filter(
                        alt.datum['Trọng số'] <= 0.05  # SỬA: Dùng bracket
                    )

                    chart = pie + text_inside + text_outside
                    st.altair_chart(chart, use_container_width=True)
                # --------------------------------------------------

                st.button(
                    f"➡️ Chuyển đến Trang Phân tích TOPSIS với mô hình '{st.session_state.model_name}'",
                    key="run_topsis_after_save",
                    on_click=switch_to_topsis_with_last_saved,
                    use_container_width=True
                )


    # --- 3. Xử lý Lựa chọn của Người dùng (Phần Kịch bản có sẵn) ---
    if selected_scenario != "--- Tạo mô hình mới ---":
        st.subheader(f"Trọng số hiện tại của mô hình: '{selected_scenario}'")

        current_weights = all_weights.get(selected_scenario, {})
        if current_weights:
            weights_df = pd.DataFrame.from_dict(current_weights, orient='index', columns=['Trọng số'])
            weights_df['%'] = (weights_df['Trọng số'] * 100).round(2).astype(str) + '%'
            st.dataframe(weights_df, use_container_width=True)
        else:
            st.warning("Mô hình này không có dữ liệu trọng số.")

        st.divider()
        st.write(f"Bạn có muốn sử dụng mô hình '{selected_scenario}' này không?")

        col1, col2, _ = st.columns([1, 1, 3])

        with col1:
            st.button(
                "Sử dụng trọng số này",
                use_container_width=True,
                on_click=switch_to_topsis_page_and_run
            )

        with col2:
            if st.button("Tùy chỉnh (Customize)", use_container_width=True):
                st.session_state.customize_mode = True
                st.session_state.selected_model_for_topsis = None
                st.session_state.last_saved_model = None
                st.session_state.last_saved_weights = None

        if st.session_state.customize_mode:
            show_customization_tabs(all_weights, model_name_placeholder=selected_scenario)

    # Xử lý cho trường hợp "Tạo mới"
    else:
        st.info("Bạn đã chọn tạo mô hình mới. Vui lòng nhập tên và thiết lập trọng số bên dưới.")
        show_customization_tabs(all_weights)


# ====================================================================
# --- TRANG 4: PHÂN TÍCH ĐỊA ĐIỂM (TOPSIS) ---
# ====================================================================
elif page == "Phân tích Địa điểm (TOPSIS)":
    st.header("Trang 4: Xếp hạng Địa điểm Tối ưu bằng TOPSIS")

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

    # --- SỬA LỖI F5 (TRANG 4) ---
    selectbox_key_topsis = "topsis_model_selector"
    default_index_topsis = 0
    model_transferred = None

    if 'selected_model_for_topsis' in st.session_state and st.session_state.selected_model_for_topsis is not None:
        model_transferred = st.session_state.selected_model_for_topsis
        if model_transferred in model_names:
            default_index_topsis = model_names.index(model_transferred)
        st.success(f"Đã tự động chọn mô hình '{model_names[default_index_topsis]}' từ Trang 3.", icon="✅")
        # Không reset state vội, để dùng cho auto-run

    elif selectbox_key_topsis in st.session_state:
        current_saved_model = st.session_state[selectbox_key_topsis]
        if current_saved_model in model_names:
            default_index_topsis = model_names.index(current_saved_model)
    # -----------------------------

    selected_model = st.selectbox(
        "Chọn một mô hình có sẵn để phân tích:",
        model_names,
        index=default_index_topsis,
        key=selectbox_key_topsis
    )


    def run_and_display_topsis(model_name):
        with st.spinner("Đang tính toán, vui lòng chờ..."):
            report_df = run_topsis_model(
                # THAY ĐỔI: Đổi tên tham số thành data_path
                data_path="AHP_Data_synced_fixed.xlsx",
                json_path="metadata.json",
                analysis_type=model_name,
                all_criteria_weights=all_weights
            )

            if report_df is not None:
                st.success("Phân tích hoàn tất!")
                st.subheader("Kết quả Xếp hạng Địa điểm")
                st.dataframe(report_df, use_container_width=True)
                st.info(
                    f"**Kết luận:** Dựa trên mô hình **{model_name.upper()}**, địa điểm tối ưu nhất là **{report_df.iloc[0]['Tên phường']}**.")

                st.divider()
                st.subheader("Hành động tiếp theo")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.button("🗺️ Xem Bản đồ (Map View)", on_click=switch_to_map_view, use_container_width=True)

                with col2:
                    st.button("Sensitivity Analysis)", on_click=switch_to_sensitivity, use_container_width=True)

                with col3:
                    st.button("⚙️ Tùy chỉnh lại Trọng số", on_click=switch_to_ahp_customize, use_container_width=True)
            else:
                st.error("Đã xảy ra lỗi trong quá trình phân tích TOPSIS. Vui lòng kiểm tra file 'metadata.json'.")


    if st.session_state.get('auto_run_topsis', False):
        st.session_state.auto_run_topsis = False

        if model_transferred in model_names:
            run_and_display_topsis(model_transferred)
            st.session_state.selected_model_for_topsis = None
        else:
            st.error("Lỗi: Không tìm thấy mô hình được chuyển. Vui lòng chọn và nhấn nút bên dưới.")
            if st.button(f"Chạy Phân tích cho mô hình '{selected_model.upper()}'"):
                run_and_display_topsis(selected_model)

    else:
        if st.button(f"Chạy Phân tích cho mô hình '{selected_model.upper()}'"):
            run_and_display_topsis(selected_model)


# ====================================================================
# --- TRANG 5: PHÂN TÍCH ĐỘ NHẠY (WHAT-IF) ---
# ====================================================================
elif page == "Phân tích Độ nhạy (What-if)":
    st.header("Trang 5: Phân tích Độ nhạy (What-if)")
    st.markdown("Thay đổi trọng số của các tiêu chí để xem kết quả xếp hạng thay đổi như thế nào so với bản gốc.")

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

    # --- SỬA LỖI F5 (TRANG 5) ---
    selectbox_key_whatif = "whatif_model_selector"
    default_index_whatif = 0

    if 'whatif_model_selector' in st.session_state and st.session_state.whatif_model_selector in model_names:
        default_index_whatif = model_names.index(st.session_state.whatif_model_selector)
    elif selectbox_key_whatif in st.session_state:
        current_saved_model_whatif = st.session_state[selectbox_key_whatif]
        if current_saved_model_whatif in model_names:
            default_index_whatif = model_names.index(current_saved_model_whatif)
    # -----------------------------

    selected_model = st.selectbox(
        "Chọn mô hình gốc để so sánh:",
        model_names,
        index=default_index_whatif,
        key=selectbox_key_whatif
    )

    if selected_model:
        original_weights = all_weights[selected_model]
        st.subheader(f"Điều chỉnh Trọng số (Mô hình: {selected_model.upper()})")

        new_weights_dict = {}

        try:
            # THAY ĐỔI: Đọc file .xlsx
            df_data = pd.read_excel("AHP_Data_synced_fixed.xlsx")
            full_criteria_list = [col for col in df_data.columns if col not in ['ward', 'ward_id']]
        except FileNotFoundError:
            st.error("Lỗi: Không tìm thấy file 'AHP_Data_synced_fixed.xlsx'.")
            st.stop()

        model_criteria = list(original_weights.keys())
        other_criteria = [c for c in full_criteria_list if c not in model_criteria]

        for criterion in model_criteria:
            new_weight = st.slider(
                f"Trọng số cho '{criterion}'",
                min_value=0.0,
                max_value=1.0,
                value=original_weights.get(criterion, 0.0),  # Sửa: dùng .get() để an toàn
                step=0.01,
                key=f"slider_{criterion}_{selected_model}"
            )
            new_weights_dict[criterion] = new_weight

        for criterion in other_criteria:
            new_weights_dict[criterion] = 0.0

        if other_criteria:
            with st.expander("Các tiêu chí không sử dụng (Trọng số = 0)"):
                st.write(other_criteria)
                # --- THÊM NÚT QUAY LẠI TÙY CHỈNH ---
                st.button(
                    "Quay lại Trang 3 để Tùy chỉnh Tiêu chí",
                    on_click=switch_to_ahp_customize,  # Dùng hàm callback đã định nghĩa
                    key="redirect_from_sensitivity"
                )
                # ------------------------------------

        total_new_weight = sum(new_weights_dict.values())
        if total_new_weight > 0:
            normalized_weights = {k: v / total_new_weight for k, v in new_weights_dict.items()}
            st.info(
                f"Tổng trọng số mới của bạn là {total_new_weight:.2f}. Kết quả sẽ được tự động chuẩn hóa về 1 để so sánh.")
        else:
            normalized_weights = new_weights_dict
            st.warning("Tất cả trọng số đều bằng 0. Kết quả sẽ không chính xác.")

        if st.button("Chạy Phân tích Độ nhạy"):
            with st.spinner("Đang chạy so sánh..."):

                original_df, new_df = run_what_if_analysis(
                    selected_model,
                    normalized_weights
                )

                if original_df is not None and new_df is not None:
                    st.success("Hoàn thành!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Xếp hạng Gốc")
                        st.dataframe(original_df, use_container_width=True, height=400)
                    with col2:
                        st.subheader("Xếp hạng Mới (What-if)")
                        st.dataframe(new_df, use_container_width=True, height=400)

                    st.divider()
                    st.subheader("Trực quan hóa sự thay đổi")


                    # --- CẬP NHẬT: LOGIC BIỂU ĐỒ MỚI ---

                    # 1. Chuẩn bị dữ liệu cho Pie Chart (gộp chung cho một legend)
                    def create_pie_data(weights_dict, title_suffix):
                        filtered_weights = {k: weights_dict.get(k, 0.0) for k in full_criteria_list if
                                            weights_dict.get(k, 0.0) > 0.001}
                        if not filtered_weights:
                            return pd.DataFrame(columns=["Tiêu chí", "Trọng số", "Loại", "Tỷ lệ"])
                        df = pd.DataFrame(filtered_weights.items(), columns=["Tiêu chí", "Trọng số"])
                        df['Loại'] = title_suffix
                        # Sửa: Tính tỷ lệ dựa trên TỔNG TRỌNG SỐ
                        df['Tỷ lệ'] = df['Trọng số'] / df['Trọng số'].sum()
                        return df


                    df_pie_original = create_pie_data(original_weights, "1. Phân bổ Gốc")
                    df_pie_new = create_pie_data(normalized_weights, "2. Phân bổ Mới")
                    df_combined_pie = pd.concat([df_pie_original, df_pie_new]).reset_index(drop=True)

                    # 2. Tạo MỘT biểu đồ duy nhất (2 cột, 1 legend)
                    st.markdown("##### So sánh Phân bổ Trọng số")

                    if not df_combined_pie.empty:
                        # --- SỬA LỖI: XẾP LỚP (LAYER) TRƯỚC, SAU ĐÓ CHIA CỘT (FACET) ---

                        # Tạo base chart
                        base = alt.Chart(df_combined_pie).encode(
                            theta=alt.Theta("Trọng số", stack=True),
                            tooltip=["Loại", "Tiêu chí", alt.Tooltip("Trọng số", format=".1%")]
                        )

                        # Lớp 1: Pie
                        pie_layer = base.mark_arc(outerRadius=120, innerRadius=0).encode(
                            color=alt.Color("Tiêu chí", title="Tiêu chí"),
                            order=alt.Order("Trọng số", sort="descending")
                        )

                        # Lớp 2: Text bên trong
                        text_inside_layer = base.mark_text(radius=80).encode(
                            text=alt.Text("Tỷ lệ", format=".1%"),
                            order=alt.Order("Trọng số", sort="descending"),
                            color=alt.value("black")
                        ).transform_filter(
                            alt.datum['Tỷ lệ'] > 0.05
                        )

                        # Lớp 3: Text bên ngoài
                        text_outside_layer = base.mark_text(radius=140).encode(
                            text=alt.Text("Tỷ lệ", format=".1%"),
                            order=alt.Order("Trọng số", sort="descending"),
                            color=alt.value("black")
                        ).transform_filter(
                            alt.datum['Tỷ lệ'] <= 0.05
                        )

                        # Kết hợp 3 LỚP lại với nhau
                        combined_layers = pie_layer + text_inside_layer + text_outside_layer

                        # Áp dụng FACET (chia cột) cho biểu đồ đã kết hợp
                        final_chart = combined_layers.facet(
                            column=alt.Column("Loại", title="Phân bổ",
                                              header=alt.Header(titleOrient="bottom", labelOrient="bottom"))
                        ).resolve_scale(
                            color='shared'  # Đảm bảo dùng 1 legend màu
                        )

                        st.altair_chart(final_chart, use_container_width=True)
                        st.caption("ℹ️ Màu sắc của các tiêu chí được giữ nguyên giữa hai biểu đồ để dễ so sánh.")
                        # -----------------------------------------------------------
                    else:
                        st.info("Không có dữ liệu trọng số để vẽ biểu đồ.")

                    # 3. Bảng thay đổi thứ hạng (nằm dưới)
                    st.markdown("##### Bảng Thay đổi Thứ hạng")

                    df_orig_simple = original_df[['Tên phường', 'Rank']].rename(columns={'Rank': 'Hạng Gốc'})
                    df_new_simple = new_df[['Tên phường', 'Rank']].rename(columns={'Rank': 'Hạng Mới'})

                    df_rank_change = pd.merge(df_orig_simple, df_new_simple, on='Tên phường')

                    df_rank_change['Thay đổi (số)'] = df_rank_change['Hạng Gốc'] - df_rank_change['Hạng Mới']


                    def format_rank_change(change):
                        if change > 0:
                            return f"🔼 +{change}"
                        elif change < 0:
                            return f"🔽 {change}"
                        else:
                            return "➖"  # Chỉ gạch ngang, không có số 0


                    df_rank_change['Thay đổi'] = df_rank_change['Thay đổi (số)'].apply(format_rank_change)

                    # --- SỬA LỖI TYPO: Sắp xếp theo 'Hạng Mới' ---
                    df_rank_change = df_rank_change.sort_values(by='Hạng Mới')

                    st.dataframe(
                        df_rank_change[['Tên phường', 'Hạng Mới', 'Hạng Gốc', 'Thay đổi']],
                        use_container_width=True,
                        hide_index=True
                    )

                else:
                    st.error(
                        "Lỗi khi chạy phân tích. Vui lòng kiểm tra file 'metadata.json' và 'topsis_module.py' (lỗi thụt lề).")


# ====================================================================
# --- TRANG 6: MAP VIEW (TRANG MỚI) ---
# ====================================================================
elif page == "Map View":
    st.header("Trang 6: Trực quan hóa Kết quả trên Bản đồ")

    model_to_map = st.session_state.get('model_for_next_page')

    if not model_to_map:
        st.warning("Vui lòng chạy một phân tích TOPSIS (ở Trang 4) trước khi xem bản đồ.")
        st.stop()

    st.success(f"Đang hiển thị kết quả cho mô hình: **{model_to_map}**")

    geojson_file = "quan7_geojson.json"
    # THAY ĐỔI: Đọc file .xlsx
    ranking_file = f"ranking_result_{model_to_map}.xlsx"

    # 2. Tải các file
    try:
        with open(geojson_file, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error(f"LỖI: Không tìm thấy file `{geojson_file}`.")
        st.markdown(
            "Vui lòng tải file GeoJSON của Quận 7 về (theo hướng dẫn) và đổi tên thành `quan7_geojson.json` rồi đặt chung thư mục với `app.py`.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi đọc file GeoJSON: {e}")
        st.stop()

    try:
        # THAY ĐỔI: Đọc file .xlsx
        df_ranking = pd.read_excel(ranking_file)
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file kết quả `{ranking_file}`.")
        st.markdown(
            f"Vui lòng quay lại **Trang 4 (Phân tích Địa điểm)** và chạy phân tích cho mô hình `{model_to_map}` ít nhất một lần để tạo file.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi đọc file excel xếp hạng: {e}")
        st.stop()

    # --- 3. SỬA LỖI: XỬ LÝ VÀ GỘP DỮ LIỆU ---

    # Tạo một dict từ df_ranking, với key đã được chuẩn hóa (xóa dấu cách)
    ranking_lookup = {}
    for _, row in df_ranking.iterrows():
        # Chuẩn hóa tên phường từ CSV (ví dụ: "Tân Thuận Tây" -> "TânThuậnTây")
        normalized_key = str(row['Tên phường']).replace(" ", "")
        ranking_lookup[normalized_key] = row.to_dict()

    max_rank = df_ranking['Rank'].max()
    missing_wards = []

    # Thêm dữ liệu (Rank, Score) vào 'properties' của GeoJSON
    for feature in geojson_data['features']:
        # Lấy tên gốc từ bản đồ
        ward_name_from_map_original = feature['properties'].get('name')

        if ward_name_from_map_original:
            # Chuẩn hóa tên phường từ GeoJSON (ví dụ: "TânThuậnTây" -> "TânThuậnTây")
            ward_name_from_map_normalized = str(ward_name_from_map_original).replace(" ", "")

            # So sánh hai tên đã được chuẩn hóa
            if ward_name_from_map_normalized in ranking_lookup:
                rank_data = ranking_lookup[ward_name_from_map_normalized]
                rank = int(rank_data['Rank'])
                score = float(rank_data['Điểm TOPSIS (0-1)'])

                # Gán thuộc tính mới
                feature['properties']['Rank'] = rank
                feature['properties']['Score'] = score

                # --- TÍNH TOÁN MÀU SẮC (Choropleth) ---
                ratio = (rank - 1) / (max_rank - 1) if max_rank > 1 else 0
                r = int(255 * ratio)
                g = int(255 * (1 - ratio))
                b = 0
                feature['properties']['color'] = [r, g, b, 180]

            else:
                # Nếu không tìm thấy (do tên không khớp)
                missing_wards.append(ward_name_from_map_original)
                feature['properties']['Rank'] = "N/A"
                feature['properties']['Score'] = "N/A"
                feature['properties']['color'] = [128, 128, 128, 100]  # Màu xám
        else:
            missing_wards.append("(Tên rỗng trong GeoJSON)")

    if missing_wards:
        st.warning(
            f"Không tìm thấy dữ liệu xếp hạng cho các phường (tên có thể không khớp): {', '.join(missing_wards)}")
        st.markdown(
            "Hãy kiểm tra `quan7_geojson.json` (key: `properties.name`) và cột `ward` trong `AHP_Data_synced_fixed.xlsx`.")

    # 4. Cấu hình Bản đồ PyDeck
    st.subheader("Bản đồ Xếp hạng TOPSIS (Choropleth Map)")
    st.markdown(
        f"Trực quan hóa cho mô hình **{model_to_map}**. Màu càng **xanh**, thứ hạng càng **cao** (Hạng 1 = Tốt nhất).")

    # THAY ĐỔI: Chuyển sang 2D (pitch=0)
    view_state = pdk.ViewState(
        latitude=10.73,
        longitude=106.72,
        zoom=13,
        pitch=0,  # <-- ĐỔI SANG 2D
        bearing=0
    )

    # THAY ĐỔI: Chuyển sang 2D (tắt extruded và elevation)
    layer = pdk.Layer(
        'GeoJsonLayer',
        geojson_data,
        opacity=0.8,
        stroked=True,
        filled=True,
        extruded=False,  # <-- TẮT 3D
        # wireframe=True, # Không cần thiết cho 2D
        get_fill_color='properties.color',
        get_line_color=[255, 255, 255],
        get_line_width=200,
        # get_elevation='properties.Score * 2000', # <-- TẮT 3D
        pickable=True,
        auto_highlight=True
    )

    tooltip = {
        "html": """
            <b>Phường:</b> {name}<br/> 
            <b>Hạng:</b> {Rank}<br/>
            <b>Điểm TOPSIS:</b> {Score}
        """,
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }

    # 5. Vẽ Bản đồ
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=pdk.map_styles.LIGHT,
        tooltip=tooltip
    )

    st.pydeck_chart(r, use_container_width=True)


