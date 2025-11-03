# app.py (UI polish + Homepage Summary)
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import json
import altair as alt
import pydeck as pdk
import re

# --- Import các module chức năng ---
try:
    from ahp_module import calculate_ahp_weights, save_weights_to_yaml
    from topsis_module import run_topsis_model
    from sensitivity_module import run_what_if_analysis
except ImportError as e:
    st.error(
        f"Lỗi import module: {e}. Vui lòng đảm bảo các file `ahp_module.py`, `topsis_module.py`, và `sensitivity_module.py` nằm cùng thư mục."
    )
    st.stop()

# --- Cấu hình trang ---
st.set_page_config(
    page_title="DSS Quận 7",
    page_icon="🦈",
    layout="wide"
)

# =============================
# Helpers: CSS + Table render
# =============================
def inject_global_css():
    st.markdown(
        """
        <style>
        .styled-table{width:100%;border-collapse:collapse!important;border-spacing:0!important;table-layout:auto;margin-bottom:24px}
        .styled-table th,.styled-table td{padding:12px 14px!important;text-align:center!important;vertical-align:middle!important}
        .fixed-height{max-height:420px;overflow:auto;margin-bottom:32px}

        /* LIGHT THEME */
        .styled-table{border:4px solid #1F2937!important;background:#FFFFFF!important}
        .styled-table thead th{font-weight:800!important;background:#F1F5F9!important;color:#0F172A!important;border:4px solid #1F2937!important}
        .styled-table tbody td{background:#FFFFFF!important;color:#0F172A!important;border:4px solid #1F2937!important}
        .styled-table tbody td:first-child{background:#F1F5F9!important;color:#0F172A!important}  /* left col = header */

        /* DARK THEME */
        @media (prefers-color-scheme: dark){
          .styled-table{border:4px solid #94A3B8!important;background:#0B1220!important}
          .styled-table thead th{background:#0E1A2B!important;color:#F8FAFC!important;border:4px solid #94A3B8!important}
          .styled-table tbody td{background:#0B1220!important;color:#E5E7EB!important;border:4px solid #94A3B8!important}
          .styled-table tbody td:first-child{background:#0E1A2B!important;color:#F8FAFC!important}  /* left col = header */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def nice_name(col: str) -> str:
    return str(col).replace("_", " ").strip().title()   # hoặc .upper()


def apply_display_names(df: pd.DataFrame, name_map: dict | None = None) -> pd.DataFrame:
    df2 = df.copy()
    if name_map:
        df2 = df2.rename(columns=name_map)
    df2.columns = [nice_name(c) for c in df2.columns]
    return df2

def add_index_col(df: pd.DataFrame, label: str = "STT") -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out.insert(0, label, range(1, len(out) + 1))
    return out

def to_html_table(df: pd.DataFrame, bold_first_col: bool = True) -> str:
    df2 = df.copy()
    df2.columns = [str(c).replace("_", " ").strip().title() for c in df2.columns]
    # Drop ward_id-like columns if present
    drop_candidates = [c for c in df2.columns
                       if str(c).strip() == "ward_id"
                       or str(c).strip().lower().replace(" ", "").replace("-", "") in {"wardid","maphuong","maward"}]
    if drop_candidates:
        df2 = df2.drop(columns=drop_candidates, errors="ignore")
    # Only prettify headers that look like code-style names
    new_cols = []
    for c in df2.columns:
        s = str(c)
        if "_" in s:
            new_cols.append(s.replace("_", " ").strip().title())
        else:
            new_cols.append(s)
    df2.columns = new_cols
    if bold_first_col and df2.shape[1] > 0:
        first = df2.columns[0]
        df2[first] = df2[first].map(lambda x: f"<strong>{x}</strong>")
    return df2.to_html(index=False, escape=False, classes="styled-table")

def display_table(df, bold_first_col=True, fixed_height=420, header_tooltips=None):
    html_tbl = to_html_table(df, bold_first_col=bold_first_col)

    if header_tooltips:
        import re as _re
        html_tbl = _re.sub(r'\sdata-tip="[^"]*"', '', html_tbl)  # gỡ tooltip cũ ở cell
        html_tbl = _inject_tooltips_on_th(html_tbl, header_tooltips)  # chỉ gắn lại ở header

    st.markdown(
        f'<div class="fixed-height" style="max-height:{fixed_height}px">{html_tbl}</div>',
        unsafe_allow_html=True
    )

def load_metadata():
    try:
        with open("data/metadata.json", "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return {}

def criteria_display_map(df_cols, meta):
    out = {}
    for c in df_cols:
        if c in ("ward", "ward_id"):
            continue
        info = meta.get(c, {})
        dn = info.get("display_name", nice_name(c))
        tp = info.get("type", "")
        label = f"{dn} ({tp.title()})" if tp else dn
        out[c] = label
    return out

def summarize_weights(weights: dict | None):
    if not weights:
        return None
    total = sum(weights.values()) or 1.0
    norm = {k: v / total for k, v in weights.items()}
    top = sorted(norm.items(), key=lambda x: x[1], reverse=True)[:5]
    return {"count": len(norm), "top": top}

def show_home_summary():
    st.subheader("Tóm tắt kết quả")
    colA, colB = st.columns([2, 3])
    with colA:
        try:
            df = pd.read_excel("data/AHP_Data_synced_fixed.xlsx")
            metadata = load_metadata()
            n_ward = int(df["ward"].nunique()) if "ward" in df.columns else len(df)
            crits = [c for c in df.columns if c not in ("ward","ward_id")]
            n_criteria = len(crits)
            types = [metadata.get(c,{}).get("type","") for c in crits]
            n_benefit = sum(1 for t in types if t=="benefit")
            n_cost = sum(1 for t in types if t=="cost")
            st.metric("Số phường", n_ward)
            st.metric("Số tiêu chí", n_criteria, help=f"Benefit: {n_benefit} · Cost: {n_cost}")
        except Exception:
            st.info("Chưa có dữ liệu để tóm tắt.")
    with colB:
        last_model = st.session_state.get("last_saved_model") or st.session_state.get("topsis_model_selector") or st.session_state.get("whatif_model_selector")
        last_weights = st.session_state.get("last_saved_weights")
        if not last_weights and last_model:
            try:
                with open("data/weights.yaml","r",encoding="utf-8") as f:
                    yw = yaml.safe_load(f) or {}
                last_weights = yw.get(last_model)
            except Exception:
                last_weights = None
        st.markdown("**AHP gần nhất**")
        if last_model and last_weights:
            st.caption(last_model)
            summary = summarize_weights(last_weights)
            if summary:
                top_items = [(nice_name(k), v) for k,v in summary["top"]]
                dfw = pd.DataFrame(top_items, columns=["Tiêu chí","Trọng số"]).reset_index(drop=True)
                dfw = add_index_col(dfw, "STT")
                display_table(dfw, bold_first_col=True, fixed_height=220)
        else:
            st.caption("Chưa có mô hình/ trọng số.")
    st.divider()
    st.markdown("**Kết quả TOPSIS gần nhất**")
    last_topsis_df = st.session_state.get("last_topsis_df")
    last_topsis_model = st.session_state.get("last_topsis_model")
    if last_topsis_df is not None and not last_topsis_df.empty:
        top3 = last_topsis_df.head(3).copy()
        top3 = add_index_col(top3, "STT")
        display_table(top3, bold_first_col=True, fixed_height=200)
        if last_topsis_model:
            st.caption(f"Mô hình: {last_topsis_model}")
    else:
        st.caption("Chưa chạy TOPSIS.")
    st.divider()
    st.markdown("**What-if gần nhất**")
    last_whatif = st.session_state.get("last_whatif_rank_changes")
    if isinstance(last_whatif, pd.DataFrame) and not last_whatif.empty:
        df_wc = last_whatif.copy()
        improved = df_wc.sort_values("Thay đổi (số)", ascending=False).head(3)
        declined = df_wc.sort_values("Thay đổi (số)", ascending=True).head(3)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Tăng hạng nhiều nhất")
            display_table(add_index_col(improved[["Tên phường","Hạng Mới","Hạng Gốc","Thay đổi"]].reset_index(drop=True),"STT"), True, 200)
        with c2:
            st.caption("Giảm hạng nhiều nhất")
            display_table(add_index_col(declined[["Tên phường","Hạng Mới","Hạng Gốc","Thay đổi"]].reset_index(drop=True),"STT"), True, 200)
    else:
        st.caption("Chưa chạy What-if.")

inject_global_css()

# --- SESSION ---
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
if 'pending_nav' not in st.session_state:
    st.session_state.pending_nav = None

def go(page_name: str):
    st.session_state.pending_nav = page_name
    st.rerun()

def switch_to_topsis_page_and_run():
    selected_scenario = st.session_state.scenario_selectbox
    st.session_state.selected_model_for_topsis = selected_scenario
    st.session_state.customize_mode = False
    st.session_state.auto_run_topsis = True
    go("Phân tích Địa điểm (TOPSIS)")
    st.session_state.last_saved_model = None
    st.session_state.last_saved_weights = None

def switch_to_topsis_with_last_saved():
    model_name = st.session_state.last_saved_model
    if model_name:
        st.session_state.selected_model_for_topsis = model_name
        st.session_state.customize_mode = False
        st.session_state.auto_run_topsis = True
        go("Phân tích Địa điểm (TOPSIS)")
        st.session_state.last_saved_model = None
        st.session_state.last_saved_weights = None

def switch_to_map_view():
    st.session_state.model_for_next_page = st.session_state.topsis_model_selector
    go("Map View")

def switch_to_sensitivity():
    st.session_state.whatif_model_selector = st.session_state.topsis_model_selector
    go("Phân tích Độ nhạy (What-if)")

def switch_to_ahp_customize():
    if st.session_state.page_navigator == "Phân tích Địa điểm (TOPSIS)":
        st.session_state.scenario_selectbox = st.session_state.topsis_model_selector
    elif st.session_state.page_navigator == "Phân tích Độ nhạy (What-if)":
        st.session_state.scenario_selectbox = st.session_state.whatif_model_selector
    st.session_state.customize_mode = True
    go("Tùy chỉnh Trọng số (AHP)")

# ================== UI NAV ==================
st.title("🦈 Hệ thống Hỗ trợ Quyết định Chọn địa điểm Quận 7")

if st.session_state.pending_nav:
    st.session_state.page_navigator = st.session_state.pending_nav
    st.session_state.pending_nav = None

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

# =============== PAGE 1: Homepage ===============
if page == "Homepage":
    st.header("Trang chủ")
    st.markdown("Sử dụng menu trái hoặc các nút dưới để chuyển trang.")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Tổng quan Dữ liệu", use_container_width=True):
            go("Tổng quan Dữ liệu")
    with c2:
        if st.button("AHP", use_container_width=True):
            go("Tùy chỉnh Trọng số (AHP)")
    with c3:
        if st.button("TOPSIS", use_container_width=True):
            go("Phân tích Địa điểm (TOPSIS)")
    d1, d2 = st.columns(2)
    with d1:
        if st.button("What-if", use_container_width=True):
            go("Phân tích Độ nhạy (What-if)")
    with d2:
        if st.button("Map View", use_container_width=True):
            go("Map View")

    st.divider()
    st.subheader("Hướng dẫn ngắn")
    st.markdown(
        """
        1) Xem dữ liệu và tiêu chí ở **Tổng quan Dữ liệu**.  
        2) Tạo hoặc chỉnh trọng số ở **AHP**.  
        3) Xếp hạng với **TOPSIS**, sau đó xem **Map View** hoặc **What-if**.
        """
    )
    show_home_summary()

# =============== PAGE 2: Data Overview ===============
elif page == "Tổng quan Dữ liệu":
    st.header("Trang 2: Khám phá và Tổng quan Dữ liệu")

    try:
        df = pd.read_excel("data/AHP_Data_synced_fixed.xlsx")
        metadata = load_metadata()
    except FileNotFoundError:
        st.error("Thiếu `AHP_Data_synced_fixed.xlsx` hoặc `metadata.json`.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        st.stop()

    name_map = {}
    for c in df.columns:
        if c in ("ward", "ward_id"):
            continue
        info = metadata.get(c, {})
        name_map[c] = info.get("display_name", nice_name(c))

    tab1, tab2 = st.tabs(["📊 Thống kê Chung", "📈 Phân tích Từng Tiêu chí"])

    with tab1:
        col1, col2 = st.columns(2)
        col1.metric("Số phường", int(df["ward"].nunique()), help="Đếm từ 1")
        col2.metric("Số tiêu chí", int(len(df.columns) - 2), help="Không tính cột ward và ward_id")

        st.subheader("Thống kê Mô tả")
        desc = df.drop(columns=["ward", "ward_id"]).describe().T.reset_index().rename(columns={"index": "Tiêu chí"})
        desc["Tiêu chí"] = desc["Tiêu chí"].map(lambda x: name_map.get(x, nice_name(x)))
        display_table(apply_display_names(desc), bold_first_col=True, fixed_height=360)

        st.subheader("Bảng Dữ liệu gốc")
        raw = df.copy().drop(columns=['ward_id'], errors='ignore')
        raw = raw.rename(columns=name_map)
        if 'ward' in raw.columns:
            raw['ward'] = raw['ward'].astype(str).str.title()  # hoặc .upper()
        raw = add_index_col(raw, "STT")
        display_table(raw, bold_first_col=True, fixed_height=420)

    with tab2:
        st.subheader("Chi tiết theo tiêu chí")
        criteria_list = [col for col in df.columns if col not in ['ward', 'ward_id']]
        cdisp_map = criteria_display_map(criteria_list, metadata)
        options = [cdisp_map[c] for c in criteria_list]
        selected_label = st.selectbox("Chọn tiêu chí:", options)
        inv_map = {v: k for k, v in cdisp_map.items()}
        selected_criterion = inv_map[selected_label]

        meta_info = metadata.get(selected_criterion, {})
        full_name = meta_info.get('display_name', nice_name(selected_criterion))
        desc = meta_info.get('description', "Không có mô tả.")
        c_type = meta_info.get('type', 'N/A')

        st.markdown(f"**{full_name}** · Loại: **{c_type.title()}**")
        st.caption(desc)
        st.divider()

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Top 5 phường")
            is_cost = (c_type == 'cost')
            sorted_df = df.sort_values(by=selected_criterion, ascending=is_cost).head(5)
            show = sorted_df[['ward', selected_criterion]].rename(columns={'ward': 'Tên phường', selected_criterion: full_name})
            show = add_index_col(show, "STT")
            display_table(show, bold_first_col=True, fixed_height=300)

        with col2:
            st.subheader("Phân phối theo phường")
            disp_df = df.drop(columns=['ward_id'], errors='ignore').rename(columns={'ward': 'Tên phường', selected_criterion: full_name})
            chart = alt.Chart(disp_df).mark_bar().encode(
                x=alt.X('Tên phường', title="Tên phường", axis=alt.Axis(labelAngle=0)),
                y=alt.Y(full_name, title=full_name),
                tooltip=['Tên phường', full_name]
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

# =============== PAGE 3: AHP Customize ===============
elif page == "Tùy chỉnh Trọng số (AHP)":
    st.header("Trang 3: Tạo và Cập nhật Trọng số Mô hình")

    all_weights = {}
    weights_file = "data/weights.yaml"
    if os.path.exists(weights_file):
        try:
            with open(weights_file, 'r', encoding='utf-8') as f:
                all_weights = yaml.safe_load(f) or {}
        except Exception as e:
            st.error(f"Lỗi khi đọc 'weights.yaml': {e}")
            all_weights = {}

    model_list = ["--- Tạo mô hình mới ---"] + list(all_weights.keys())
    st.subheader("1. Lựa chọn Kịch bản (Scenario)")

    selectbox_key_ahp = "scenario_selectbox"
    default_index_ahp = 0
    if 'scenario_selectbox' in st.session_state and st.session_state.scenario_selectbox in model_list:
        default_index_ahp = model_list.index(st.session_state.scenario_selectbox)

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

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Tắt Customize", use_container_width=True):
            st.session_state.customize_mode = False
            go("Phân tích Địa điểm (TOPSIS)")
    with c2:
        st.caption("Mặc định hiển thị Pie chart sau khi có trọng số.")

    def show_customization_tabs(all_weights_passed_in, model_name_placeholder=""):
        metadata = load_metadata()
        try:
            df_data = pd.read_excel("data/AHP_Data_synced_fixed.xlsx")
            full_criteria_list = [col for col in df_data.columns if col not in ['ward', 'ward_id']]
        except FileNotFoundError:
            st.error("Thiếu file dữ liệu.")
            st.stop()

        cdisp_map = criteria_display_map(full_criteria_list, metadata)

        if model_name_placeholder:
            st.subheader(f"2. Tùy chỉnh Trọng số cho: **{model_name_placeholder}**")
            st.session_state.model_name = model_name_placeholder
        else:
            st.subheader("2. Tùy chỉnh Trọng số")
            st.session_state.model_name = st.text_input("Nhập tên mô hình mới:")

        if not st.session_state.model_name:
            return

        st.divider()
        st.subheader("2.5 Chọn Tiêu chí sử dụng")
        original_weights_dict = all_weights_passed_in.get(st.session_state.model_name, {})
        default_selection = list(original_weights_dict.keys()) if original_weights_dict else full_criteria_list

        cols = st.columns(3)
        selected_criteria_list = []
        for i, criterion in enumerate(full_criteria_list):
            label = cdisp_map.get(criterion, nice_name(criterion))
            is_checked = criterion in default_selection
            with cols[i % 3]:
                if st.checkbox(
                    label,
                    value=is_checked,
                    key=f"check_{criterion}_{st.session_state.model_name}"
                ):
                    selected_criteria_list.append(criterion)

        st.divider()
        if not selected_criteria_list:
            st.warning("Chọn ít nhất một tiêu chí.")
            st.stop()

        tab1, tab2 = st.tabs(
            ["Phương pháp 1: Điểm 1–10", "Phương pháp 2: Ma trận so sánh cặp (AHP)"]
        )

        with tab1:
            st.info("Gán điểm 1–10 cho từng tiêu chí. Tự động chuẩn hóa thành trọng số.")
            scores_dict = {}
            if not original_weights_dict:
                scores_dict = {c: 5 for c in selected_criteria_list}
            else:
                max_weight = max(original_weights_dict.values()) if original_weights_dict else 1
                if max_weight == 0:
                    max_weight = 1
                scores_dict = {k: int(round((v / max_weight) * 9 + 1)) for k, v in original_weights_dict.items()
                               if k in selected_criteria_list}
                for c in selected_criteria_list:
                    scores_dict.setdefault(c, 5)

            new_scores = {}
            for c in selected_criteria_list:
                label = cdisp_map.get(c, nice_name(c))
                score = st.slider(
                    label,
                    min_value=1,
                    max_value=10,
                    value=scores_dict.get(c, 5),
                    key=f"score_{c}_{st.session_state.model_name}"
                )
                new_scores[c] = score

            total_score = sum(new_scores.values())
            if total_score > 0:
                normalized_weights = {k: v / total_score for k, v in new_scores.items()}

                st.subheader("Trọng số (chuẩn hóa)")
                dfw = pd.DataFrame(
                    [(cdisp_map.get(k, nice_name(k)), v) for k, v in normalized_weights.items()],
                    columns=["Tiêu chí", "Trọng số"]
                ).sort_values("Trọng số", ascending=False).reset_index(drop=True)
                dfw = add_index_col(dfw, "STT")
                display_table(dfw, bold_first_col=True, fixed_height=300)

                base = alt.Chart(dfw.rename(columns={"Trọng số":"Weight"})).encode(theta=alt.Theta("Weight", stack=True))
                pie = base.mark_arc(outerRadius=120).encode(color="Tiêu chí", tooltip=["Tiêu chí", alt.Tooltip("Weight", format=".1%")])
                inside = base.mark_text(radius=80).encode(text=alt.Text("Weight", format=".1%")).transform_filter(alt.datum.Weight > 0.05)
                outside = base.mark_text(radius=140).encode(text=alt.Text("Weight", format=".1%")).transform_filter(alt.datum.Weight <= 0.05)
                st.altair_chart(pie + inside + outside, use_container_width=True)

                if st.button("Lưu Trọng số (Phương pháp 1)", key="save_method_1"):
                    saved_ok = save_weights_to_yaml(normalized_weights, st.session_state.model_name)
                    if saved_ok:
                        st.session_state.last_saved_model = st.session_state.model_name
                        st.session_state.last_saved_weights = normalized_weights
                        st.rerun()
                    else:
                        st.error("Không thể lưu.")
            else:
                st.warning("Tổng điểm bằng 0.")

        with tab2:
            st.info("Nhập ma trận so sánh cặp. Giá trị 1–9. CR < 0.1 được chấp nhận.")
            n = len(selected_criteria_list)
            matrix_state_key = f"ahp_matrix_{st.session_state.model_name}_{'_'.join(sorted(selected_criteria_list))}"
            if (matrix_state_key not in st.session_state.ahp_matrices or
                    st.session_state.ahp_matrices[matrix_state_key].shape[0] != n):
                st.session_state.ahp_matrices[matrix_state_key] = np.ones((n, n))
            current_matrix = st.session_state.ahp_matrices[matrix_state_key]

            header_cols = st.columns([1.5] + [1] * n)
            for j, col_name in enumerate(selected_criteria_list):
                with header_cols[j + 1]:
                    st.write(f"**{cdisp_map.get(col_name, nice_name(col_name))}**")

            for i in range(n):
                row_cols = st.columns([1.5] + [1] * n)
                with row_cols[0]:
                    st.write("")
                    st.write(f"**{cdisp_map.get(selected_criteria_list[i], nice_name(selected_criteria_list[i]))}**")
                for j in range(n):
                    with row_cols[j + 1]:
                        if i == j:
                            st.text_input(f"diag_{i}_{j}", value="1.00", disabled=True, label_visibility="collapsed")
                        elif i < j:
                            key = f"matrix_{i}_{j}_{matrix_state_key}"
                            val = st.session_state.get(key, 1.0)
                            current_matrix[i, j] = val
                            if val != 0:
                                current_matrix[j, i] = 1.0 / val
                            st.number_input(label=f"Input {i}-{j}", min_value=0.01, value=current_matrix[i, j],
                                            step=0.1, format="%.2f", label_visibility="collapsed", key=key)
                        else:
                            st.text_input(f"low_{i}_{j}", value=f"{current_matrix[i, j]:.2f}",
                                          disabled=True, label_visibility="collapsed")

            if st.button("Tính toán và Lưu Trọng số (Phương pháp 2)", key="save_method_2"):
                weights, cr = calculate_ahp_weights(current_matrix)
                if weights is not None and cr is not None and cr < 0.1:
                    st.success(f"CR = {cr:.4f}")
                    weights_dict = {name: weight for name, weight in zip(selected_criteria_list, weights)}
                    saved_ok = save_weights_to_yaml(weights_dict, st.session_state.model_name)
                    if saved_ok:
                        st.session_state.last_saved_model = st.session_state.model_name
                        st.session_state.last_saved_weights = weights_dict
                        st.rerun()
                    else:
                        st.error("Không thể lưu.")
                else:
                    st.error(f"CR không đạt: {cr if cr is not None else 'N/A'}")

    if selected_scenario != "--- Tạo mô hình mới ---":
        st.subheader(f"Trọng số hiện tại: **{selected_scenario}**")
        current_weights = all_weights.get(selected_scenario, {})
        if current_weights:
            dfw = pd.DataFrame(
                [(nice_name(k), v) for k, v in current_weights.items()],
                columns=["Tiêu chí", "Trọng số"]
            ).sort_values("Trọng số", ascending=False).reset_index(drop=True)
            dfw = add_index_col(dfw, "STT")
            display_table(dfw, bold_first_col=True, fixed_height=300)
        else:
            st.warning("Mô hình này chưa có trọng số.")

        st.divider()
        col1, col2, _ = st.columns([1,1,3])
        with col1:
            st.button("Sử dụng trọng số này", use_container_width=True, on_click=switch_to_topsis_page_and_run)
        with col2:
            if st.button("Tùy chỉnh (Customize)", use_container_width=True):
                st.session_state.customize_mode = True
                st.session_state.selected_model_for_topsis = None
                st.session_state.last_saved_model = None
                st.session_state.last_saved_weights = None

        if st.session_state.customize_mode:
            show_customization_tabs(all_weights, model_name_placeholder=selected_scenario)

    else:
        st.info("Tạo mô hình mới.")
        show_customization_tabs(all_weights)

    if (st.session_state.get('last_saved_model') == st.session_state.get('model_name') and
            st.session_state.get('last_saved_weights')):
        st.divider()
        weights_dict = st.session_state.last_saved_weights
        df_chart = pd.DataFrame(weights_dict.items(), columns=["Tiêu chí(gốc)", "Trọng số"])
        df_chart["Tiêu chí"] = df_chart["Tiêu chí(gốc)"].map(nice_name)
        df_chart = df_chart.drop(columns=["Tiêu chí(gốc)"])
        base = alt.Chart(df_chart).encode(theta=alt.Theta("Trọng số", stack=True))
        pie = base.mark_arc(outerRadius=120).encode(color="Tiêu chí", tooltip=["Tiêu chí", alt.Tooltip("Trọng số", format=".1%")])
        inside = base.mark_text(radius=80).encode(text=alt.Text("Trọng số", format=".1%")).transform_filter(alt.datum["Trọng số"] > 0.05)
        outside = base.mark_text(radius=140).encode(text=alt.Text("Trọng số", format=".1%")).transform_filter(alt.datum["Trọng số"] <= 0.05)
        st.altair_chart(pie + inside + outside, use_container_width=True)

        st.button(
            f"➡️ Chuyển đến TOPSIS với '{st.session_state.model_name}'",
            key="run_topsis_after_save",
            on_click=switch_to_topsis_with_last_saved,
            use_container_width=True
        )

# =============== PAGE 4: TOPSIS ===============
elif page == "Phân tích Địa điểm (TOPSIS)":
    st.header("Trang 4: Xếp hạng Địa điểm TOPSIS")

    try:
        with open("data/weights.yaml", 'r', encoding='utf-8') as f:
            all_weights = yaml.safe_load(f) or {}
            model_names = list(all_weights.keys())
            if not model_names:
                st.warning("Chưa có mô hình AHP.")
                st.stop()
    except FileNotFoundError:
        st.error("Thiếu 'weights.yaml'.")
        st.stop()

    selectbox_key_topsis = "topsis_model_selector"
    default_index_topsis = 0
    model_transferred = None

    if 'selected_model_for_topsis' in st.session_state and st.session_state.selected_model_for_topsis is not None:
        model_transferred = st.session_state.selected_model_for_topsis
        if model_transferred in model_names:
            default_index_topsis = model_names.index(model_transferred)
        st.success(f"Tự động chọn mô hình '{model_names[default_index_topsis]}'")
    elif selectbox_key_topsis in st.session_state:
        current_saved_model = st.session_state[selectbox_key_topsis]
        if current_saved_model in model_names:
            default_index_topsis = model_names.index(current_saved_model)

    selected_model = st.selectbox(
        "Chọn mô hình:",
        model_names,
        index=default_index_topsis,
        key=selectbox_key_topsis
    )

    def run_and_display_topsis(model_name):
        st.session_state['last_topsis_model'] = model_name
        report_df = run_topsis_model(
            data_path="data/AHP_Data_synced_fixed.xlsx",
            json_path="data/metadata.json",
            analysis_type=model_name,
            all_criteria_weights=all_weights
        )
        if report_df is not None:
            st.session_state['last_topsis_df'] = report_df.copy()
            st.subheader("Kết quả xếp hạng")
            show = report_df.copy()
            show = add_index_col(show, "STT")
            display_table(show, bold_first_col=True, fixed_height=420)

            st.divider()
            cols = st.columns(3)
            with cols[0]:
                st.button("Map View", on_click=switch_to_map_view, use_container_width=True)
            with cols[1]:
                st.button("Sensitivity", on_click=switch_to_sensitivity, use_container_width=True)
            with cols[2]:
                st.button("Customize AHP", on_click=switch_to_ahp_customize, use_container_width=True)
        else:
            st.error("Lỗi khi phân tích TOPSIS.")

    if st.session_state.get('auto_run_topsis', False):
        st.session_state.auto_run_topsis = False
        if model_transferred in model_names:
            run_and_display_topsis(model_transferred)
            st.session_state.selected_model_for_topsis = None
        else:
            st.error("Không tìm thấy mô hình được chuyển.")
            if st.button(f"Chạy '{selected_model.upper()}'"):
                run_and_display_topsis(selected_model)
    else:
        if st.button(f"Chạy '{selected_model.upper()}'"):
            run_and_display_topsis(selected_model)

# =============== PAGE 5: What-if ===============
elif page == "Phân tích Độ nhạy (What-if)":
    st.header("Trang 5: Phân tích Độ nhạy (What-if)")

    try:
        with open("data/weights.yaml", 'r', encoding='utf-8') as f:
            all_weights = yaml.safe_load(f) or {}
            model_names = list(all_weights.keys())
            if not model_names:
                st.warning("Chưa có mô hình AHP.")
                st.stop()
    except FileNotFoundError:
        st.error("Thiếu 'weights.yaml'.")
        st.stop()

    selectbox_key_whatif = "whatif_model_selector"
    default_index_whatif = 0
    if 'whatif_model_selector' in st.session_state and st.session_state.whatif_model_selector in model_names:
        default_index_whatif = model_names.index(st.session_state.whatif_model_selector)

    selected_model = st.selectbox(
        "Chọn mô hình gốc:",
        model_names,
        index=default_index_whatif,
        key=selectbox_key_whatif
    )

    if selected_model:
        original_weights = all_weights[selected_model]
        st.subheader(f"Điều chỉnh Trọng số — {selected_model.upper()}")

        new_weights_dict = {}
        try:
            df_data = pd.read_excel("data/AHP_Data_synced_fixed.xlsx")
            full_criteria_list = [c for c in df_data.columns if c not in ["ward", "ward_id"]]
        except FileNotFoundError:
            st.error("Thiếu dữ liệu.")
            st.stop()

        model_criteria = list(original_weights.keys())
        other_criteria = [c for c in full_criteria_list if c not in model_criteria]

        for criterion in model_criteria:
            new_weight = st.slider(
                f"{nice_name(criterion)}",
                min_value=0.0,
                max_value=1.0,
                value=original_weights.get(criterion, 0.0),
                step=0.01,
                key=f"slider_{criterion}_{selected_model}"
            )
            new_weights_dict[criterion] = new_weight

        if other_criteria:
            st.markdown("**Tiêu chí không sử dụng (trọng số = 0)**")
            df_unused = pd.DataFrame({"Tiêu chí không sử dụng": [nice_name(c) for c in other_criteria]})
            df_unused = add_index_col(df_unused, "STT")
            display_table(df_unused, bold_first_col=True, fixed_height=240)

        total_new_weight = sum(new_weights_dict.values())
        normalized_weights = {k: (v / total_new_weight if total_new_weight > 0 else 0.0) for k, v in new_weights_dict.items()}
        if total_new_weight > 0:
            st.caption(f"Tổng trọng số mới = {total_new_weight:.2f}. Đã chuẩn hóa trước khi so sánh.")
        else:
            st.warning("Tất cả trọng số đều bằng 0.")

        if st.button("Chạy What-if"):
            original_df, new_df = run_what_if_analysis(selected_model, normalized_weights)
            if original_df is not None and new_df is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Xếp hạng gốc")
                    display_table(add_index_col(original_df.copy(), "STT"), bold_first_col=True, fixed_height=420)
                with col2:
                    st.subheader("Xếp hạng mới")
                    display_table(add_index_col(new_df.copy(), "STT"), bold_first_col=True, fixed_height=420)

                st.divider()
                st.subheader("So sánh phân bổ trọng số")

                def create_pie_data(weights_dict, title_suffix):
                    if not weights_dict:
                        return pd.DataFrame(columns=["Tiêu chí", "Trọng số", "Loại", "Tỷ lệ"])
                    dfw = pd.DataFrame(list(weights_dict.items()), columns=["Tiêu chí", "Trọng số"])
                    dfw["Loại"] = title_suffix
                    s = dfw["Trọng số"].sum()
                    dfw["Tỷ lệ"] = dfw["Trọng số"] / (s if s > 0 else 1)
                    dfw["Tiêu chí"] = dfw["Tiêu chí"].map(nice_name)
                    return dfw

                df_pie_original = create_pie_data(original_weights, "1. Gốc")
                df_pie_new = create_pie_data(normalized_weights, "2. Mới")
                df_combined = pd.concat([df_pie_original, df_pie_new], ignore_index=True)

                if not df_combined.empty:
                    base = alt.Chart(df_combined).encode(theta=alt.Theta("Trọng số", stack=True))
                    pie = base.mark_arc(outerRadius=120).encode(color=alt.Color("Tiêu chí"), tooltip=["Loại", "Tiêu chí", alt.Tooltip("Trọng số", format=".1%")])
                    t_in = base.mark_text(radius=80).encode(text=alt.Text("Tỷ lệ", format=".1%")).transform_filter(alt.datum["Tỷ lệ"] > 0.05)
                    t_out = base.mark_text(radius=140).encode(text=alt.Text("Tỷ lệ", format=".1%")).transform_filter(alt.datum["Tỷ lệ"] <= 0.05)
                    chart = (pie + t_in + t_out).facet(column=alt.Column("Loại", title="Phân bổ"))
                    st.altair_chart(chart, use_container_width=True)

                st.subheader("Bảng thay đổi thứ hạng")
                df_orig_simple = original_df[['Tên phường', 'Rank']].rename(columns={'Rank': 'Hạng Gốc'})
                df_new_simple = new_df[['Tên phường', 'Rank']].rename(columns={'Rank': 'Hạng Mới'})
                df_rank_change = pd.merge(df_orig_simple, df_new_simple, on='Tên phường')
                df_rank_change['Thay đổi (số)'] = df_rank_change['Hạng Gốc'] - df_rank_change['Hạng Mới']

                def fmt(change):
                    if change > 0:
                        return f"▲ +{change}"
                    elif change < 0:
                        return f"▼ {change}"
                    else:
                        return "—"
                df_rank_change['Thay đổi'] = df_rank_change['Thay đổi (số)'].apply(fmt)
                df_rank_change = df_rank_change.sort_values(by='Hạng Mới')
                st.session_state['last_whatif_rank_changes'] = df_rank_change.copy()
                display_table(df_rank_change[['Tên phường', 'Hạng Mới', 'Hạng Gốc', 'Thay đổi']], bold_first_col=True, fixed_height=420)
            else:
                st.error("Lỗi khi chạy What-if.")

# =============== PAGE 6: Map View ===============
elif page == "Map View":
    st.header("Trang 6: Trực quan bản đồ")

    model_to_map = st.session_state.get('model_for_next_page')
    if not model_to_map:
        st.warning("Cần chạy TOPSIS trước để chọn mô hình.")
        st.stop()

    st.success(f"Kết quả cho mô hình: **{model_to_map}**")

    geojson_file = "data/quan7_geojson.json"
    ranking_file = f"data/ranking_result_{model_to_map}.xlsx"

    try:
        with open(geojson_file, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Thiếu `{geojson_file}`.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi đọc GeoJSON: {e}")
        st.stop()

    try:
        df_ranking = pd.read_excel(ranking_file)
    except FileNotFoundError:
        st.error(f"Thiếu `{ranking_file}`. Hãy chạy TOPSIS cho mô hình này trước.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi đọc file xếp hạng: {e}")
        st.stop()

    ranking_lookup = {}
    for _, row in df_ranking.iterrows():
        normalized_key = str(row['Tên phường']).replace(" ", "")
        ranking_lookup[normalized_key] = row.to_dict()

    max_rank = df_ranking['Rank'].max()
    missing_wards = []

    def color_from_ratio(ratio: float):
        if ratio <= 0.5:
            t = ratio / 0.5
            r = int(0 + t * (255 - 0))
            g = int(170 + t * (204 - 170))
            b = int(85 - t * (85 - 0))
        else:
            t = (ratio - 0.5) / 0.5
            r = int(255 - t * (255 - 204))
            g = int(204 - t * 204)
            b = 0
        return [r, g, b, 200]

    for feature in geojson_data['features']:
        ward_name_from_map_original = feature['properties'].get('name')
        if ward_name_from_map_original:
            ward_name_from_map_normalized = str(ward_name_from_map_original).replace(" ", "")
            if ward_name_from_map_normalized in ranking_lookup:
                rank_data = ranking_lookup[ward_name_from_map_normalized]
                rank = int(rank_data['Rank'])
                score = float(rank_data['Điểm TOPSIS (0-1)'])
                feature['properties']['Rank'] = rank
                feature['properties']['Score'] = score
                ratio = (rank - 1) / (max_rank - 1) if max_rank > 1 else 0
                feature['properties']['color'] = color_from_ratio(ratio)
            else:
                missing_wards.append(ward_name_from_map_original)
                feature['properties']['Rank'] = "N/A"
                feature['properties']['Score'] = "N/A"
                feature['properties']['color'] = [128, 128, 128, 120]
        else:
            missing_wards.append("(Tên rỗng)")

    if missing_wards:
        st.warning("Tên phường không khớp: " + ", ".join(missing_wards))

    st.subheader("Bản đồ Xếp hạng TOPSIS")
    st.caption("Xanh tốt hơn. Viền đen.")

    view_state = pdk.ViewState(
        latitude=10.73, longitude=106.72, zoom=13, pitch=0, bearing=0
    )
    layer = pdk.Layer(
        'GeoJsonLayer',
        geojson_data,
        opacity=0.85,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color='properties.color',
        get_line_color=[0, 0, 0],
        get_line_width=300,
        pickable=True,
        auto_highlight=True
    )
    tooltip = {
        "html": """
            <b>Phường:</b> {name}<br/> 
            <b>Hạng:</b> {Rank}<br/>
            <b>Điểm TOPSIS:</b> {Score}
        """,
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style=pdk.map_styles.LIGHT, tooltip=tooltip)
    st.pydeck_chart(r, use_container_width=True)
