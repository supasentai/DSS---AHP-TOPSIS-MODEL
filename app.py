# app.py (UI polish + Homepage Summary)
import streamlit as st
from pathlib import Path


# === PATH NORMALIZATION ===
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

def F(name: str) -> str:
    """Return absolute path under ./data for reading (fallback to root if missing)."""
    p = DATA_DIR / name
    if p.exists():
        return str(p)
    q = ROOT / name
    return str(p if p.exists() else q)

def FW(name: str) -> str:
    """Return absolute path under ./data for writing; ensure parents exist."""
    p = DATA_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)

import pandas as pd
import re
from html import escape as _esc
import numpy as np
import yaml
import os
import json
import altair as alt
import pydeck as pdk

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
        
/* Tooltip header-only */
.styled-table th[data-tip]{ position:relative; overflow:visible; }
.styled-table th[data-tip]:hover::before{
  content:"";
  position:absolute;
  left:50%; top:calc(100% + 2px);
  transform:translateX(-50%);
  border:6px solid transparent;
  border-bottom-color: rgba(15,15,20,.98);
  z-index:99998; pointer-events:none;
}
.styled-table th[data-tip]:hover::after{
  content: attr(data-tip);
  position:absolute;
  left:50%; top:calc(100% + 14px);
  transform:translateX(-50%);
  z-index:99999; background: rgba(15,15,20,.98); color:#fff;
  padding:12px 14px; border-radius:10px; border:1px solid rgba(255,255,255,.12);
  display:block; width:max-content; min-width:16ch; max-width:min(68ch, 80vw);
  white-space:normal; word-break:normal; overflow-wrap:break-word;
  line-height:1.35rem; font-size:.95rem; box-shadow:0 10px 26px rgba(0,0,0,.40);
  pointer-events:none;
}
/* Disable tooltips in data cells */
.styled-table td[data-tip],
.styled-table td [data-tip]{ position:static; }
.styled-table td[data-tip]::before,
.styled-table td[data-tip]::after,
.styled-table td [data-tip]::before,
.styled-table td [data-tip]::after{
  content:none !important; display:none !important;
}

</style>
        """,
        unsafe_allow_html=True
    )

def nice_name(col: str) -> str:
    return str(col).replace("_", " ").strip().title()   # hoặc .upper()

def _next_clone_name(base_name, existing_names):
    base = str(base_name).strip() or "Custom"
    ex = {str(x).strip().lower() for x in existing_names}
    i = 1
    cand = f"{base}_{i}"
    while cand.strip().lower() in ex:
        i += 1
        cand = f"{base}_{i}"
    return cand

def _load_defaultweights_all(path="data/defaultweights.yaml"):
    import yaml, os
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return d if isinstance(d, dict) else {}

def _weights_equal(a: dict, b: dict, tol=1e-9):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a.keys():
        try:
            if abs(float(a[k]) - float(b[k])) > tol:
                return False
        except Exception:
            return False
    return True

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

def _inject_tooltips_on_th(html_table: str, header_tooltips: dict) -> str:
    if not header_tooltips:
        return html_table
    m = re.search(r'<thead[^>]*>.*?<tr[^>]*>(.*?)</tr>.*?</thead>', html_table, flags=re.S|re.I)
    if not m:
        return html_table
    head_row = m.group(1)
    ths = list(re.finditer(r'<th\b[^>]*>(.*?)</th>', head_row, flags=re.S|re.I))

    def _norm(s: str) -> str:
        s = re.sub(r'<[^>]+>', '', str(s))
        return re.sub(r'\\s+', ' ', s).strip().lower()

    tips = { _norm(k): v for k, v in header_tooltips.items() }
    new_cells = []
    for thm in ths:
        cell  = thm.group(0)
        label = _norm(thm.group(1))
        tip   = tips.get(label)
        if tip and 'data-tip=' not in cell:
            esc = _esc(str(tip), quote=True)
            cell = re.sub(r'(<th\b[^>]*)>', '\\1 data-tip="' + esc + '">', cell, flags=re.I)
        new_cells.append(cell)
    new_head = ''.join(new_cells)
    return html_table[:m.start(1)] + new_head + html_table[m.end(1):]


def display_table(df, bold_first_col=True, fixed_height=420, header_tooltips=None):
    html_tbl = to_html_table(df, bold_first_col=bold_first_col)

    # Ensure CSS class for tooltip selectors
    if '<table' in html_tbl:
        open_tag = html_tbl.split('>', 1)[0]
        if 'class=' not in open_tag:
            html_tbl = html_tbl.replace('<table', '<table class="styled-table"', 1)
        elif 'styled-table' not in open_tag:
            html_tbl = html_tbl.replace('class="', 'class="styled-table ', 1)

    # Remove any data-tip remnants (avoid tooltips on <td>)
    html_tbl = re.sub(r'\sdata-tip="[^"]*"', '', html_tbl)

    # Header-only tooltips
    if header_tooltips:
        html_tbl = _inject_tooltips_on_th(html_tbl, header_tooltips)

    st.markdown(
        f'<div class="fixed-height" style="{("" if fixed_height is None else f"max-height:{int(fixed_height)}px;overflow:auto;")}">{html_tbl}</div>',
        unsafe_allow_html=True
    )

def load_metadata():
    try:
        with open(F("metadata.json"), "r", encoding="utf-8-sig") as f:
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
            df = pd.read_excel(F("AHP_Data_synced_fixed.xlsx"))
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
                with open(F("weights.yaml"),"r",encoding="utf-8") as f:
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
        df = pd.read_excel(F("AHP_Data_synced_fixed.xlsx"))
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

        def _resolve_desc_tooltips(df):
            BASE = {
                'count': "Số bản ghi hợp lệ (không tính NaN).",
                'mean' : "Trung bình số học.",
                'std'  : "Độ lệch chuẩn mẫu (ddof=1).",
                'min'  : "Giá trị nhỏ nhất.",
                '25%'  : "Phân vị 25 (Q1).",
                '50%'  : "Phân vị 50 (Median).",
                '75%'  : "Phân vị 75 (Q3).",
                'max'  : "Giá trị lớn nhất."
            }
            ALIAS = {
                'count': ['count','số lượng','số bản ghi','số mẫu'],
                'mean' : ['mean','trung bình','giá trị trung bình'],
                'std'  : ['std','độ lệch chuẩn','đlc'],
                'min'  : ['min','nhỏ nhất','thấp nhất'],
                '25%'  : ['25%','q1','phân vị 25'],
                '50%'  : ['50%','median','trung vị','phân vị 50'],
                '75%'  : ['75%','q3','phân vị 75'],
                'max'  : ['max','lớn nhất','cao nhất'],
            }
            def norm(s):
                s = re.sub(r'<[^>]+>', '', str(s))
                return re.sub(r'\s+',' ', s).strip().lower()
            tips = {}
            for col in df.columns:
                c = norm(col)
                for k, names in ALIAS.items():
                    if c == k or any(c == norm(n) for n in names):
                        tips[str(col)] = BASE[k]
                        break
            return tips
    

    st.subheader("Thống kê Mô tả")
    desc = df.drop(columns=["ward", "ward_id"]).describe().T.reset_index().rename(columns={"index": "Tiêu chí"})
    desc["Tiêu chí"] = desc["Tiêu chí"].map(lambda x: name_map.get(x, nice_name(x)))
    _desc_view = apply_display_names(desc)
    display_table(_desc_view, bold_first_col=True, fixed_height=360, header_tooltips=_resolve_desc_tooltips(_desc_view))

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
    weights_file = "weights.yaml"
    if os.path.exists(weights_file):
        try:
            with open(weights_file, "r", encoding="utf-8") as f:
                all_weights = yaml.safe_load(f) or {}
        except Exception as e:
            st.error(f"Lỗi khi đọc 'weights.yaml': {e}")
            all_weights = {}

    model_list = ["Tạo mô hình mới", "Office", "Warehouse", "Factory"]
    st.subheader("1. Lựa chọn Kịch bản (Scenario)")

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
        key="scenario_selectbox",
        on_change=on_scenario_change
    )

    def _load_default_weights():
        paths = [F("defaultweights.yaml"), "defaultweights.yaml"]
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = yaml.safe_load(f)
                    if isinstance(obj, dict):
                        return {str(k).lower(): v for k, v in obj.items() if isinstance(v, dict)}
            except Exception:
                continue
        return {}

    def save_user_weights_to_yaml(weights_dict: dict, model_name: str):
        path = F("defaultweights.yaml")
        try:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except FileNotFoundError:
                data = {}
            if not isinstance(data, dict):
                data = {}
            data[str(model_name)] = weights_dict
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            return True
        except Exception as e:
            st.error(f"Lỗi lưu defaultweights.yaml: {e}")
            return False


    def quick_customize_editor(current_weights: dict, scenario_name: str):
        st.write("Tùy chỉnh nhanh: chỉnh sửa trọng số rồi lưu.")

        # Bên trái = Tiêu chí (str), bên phải = Trọng số (float trong [0,1])
        df_init = pd.DataFrame(
            [(str(k), current_weights.get(k)) for k in (current_weights or {}).keys()],
            columns=["Tiêu chí", "Trọng số"]
        )

        # Gợi ý tên theo ngữ cảnh
        try:
            existing_names = list(_load_defaultweights_all().keys())
        except Exception:
            existing_names = []
        base_defaults = {"office", "warehouse", "factory"}
        default_name = (
            _next_clone_name(scenario_name, existing_names)
            if scenario_name.strip().lower() in base_defaults
            else (scenario_name or _next_clone_name("Custom", existing_names))
        )
        model_name = st.text_input("Tên kịch bản", value=str(default_name), key=f"name_{scenario_name}")

        ed = st.data_editor(
            df_init,
            column_order=["Tiêu chí", "Trọng số"],
            column_config={
                "Tiêu chí": st.column_config.TextColumn("Tiêu chí"),
                "Trọng số": st.column_config.NumberColumn(
                    "Trọng số", min_value=0.0, max_value=1.0, step=0.01, format="%.4f"
                ),
            },
            num_rows="dynamic", hide_index=True, use_container_width=True,
            key=f"quick_edit_{scenario_name}"
        )

        # Validate: tên không rỗng, số thực trong [0,1]
        _tmp = ed.copy()
        _tmp["Tiêu chí"] = _tmp["Tiêu chí"].astype(str).str.strip()
        _tmp["Trọng số"] = pd.to_numeric(_tmp["Trọng số"], errors="coerce")

        if _tmp.empty:
            invalid_quick = True
            edited = {}
        else:
            valid_name = _tmp["Tiêu chí"].ne("")
            valid_range = _tmp["Trọng số"].between(0.0, 1.0, inclusive="both")
            invalid_quick = bool(
                _tmp["Trọng số"].isna().any() or (~valid_name).any() or (~valid_range).any()
            )
            _valid = _tmp.loc[valid_name & valid_range & _tmp["Trọng số"].notna()].copy()
            _valid["Trọng số"] = _valid["Trọng số"].astype(float)
            edited = dict(zip(_valid["Tiêu chí"], _valid["Trọng số"]))

        # Hai nút cùng hàng
        c1, c2 = st.columns(2)

        with c1:
            disabled = bool(invalid_quick or (len(edited) == 0))
            if st.button("Lưu bộ tuỳ chỉnh (defaultweights.yaml)", use_container_width=True, disabled=disabled,
                         key=f"btn_save_{scenario_name}"):
                # 1) Không lưu nếu không có thay đổi so với current_weights
                if _weights_equal(edited, current_weights or {}):
                    st.info("Không có thay đổi so với kịch bản gốc. Bỏ qua lưu.")
                else:
                    # 2) Chuẩn hoá tên đích, tránh overwrite bản gốc
                    try:
                        exists = _load_defaultweights_all()
                    except Exception:
                        exists = {}
                    names = list(exists.keys()) if isinstance(exists, dict) else []
                    target = (model_name or "").strip() or _next_clone_name(scenario_name, names)

                    # Không được ghi đè Office/Warehouse/Factory
                    if target.strip().lower() in base_defaults:
                        target = _next_clone_name(target, names)

                    # Nếu tên đã tồn tại
                    if target in names:
                        # Trùng nội dung thì bỏ qua, khác nội dung thì clone sang tên mới
                        if isinstance(exists, dict) and _weights_equal(exists.get(target, {}), edited):
                            st.info("Nội dung trùng bản hiện có. Không lưu mới.")
                            st.stop()
                        target = _next_clone_name(target, names)

                    ok = save_user_weights_to_yaml(edited, target)
                    if ok:
                        st.success(f"Đã lưu '{target}' vào defaultweights.yaml")

        with c2:
            disabled = bool(invalid_quick or (len(edited) == 0))
            if st.button("Tiếp tục qua trang phân tích", use_container_width=True, disabled=disabled,
                         key=f"btn_next_{scenario_name}"):
                st.session_state["selected_model_for_topsis"] = scenario_name
                st.session_state["selected_weights_for_topsis"] = edited
                go("Phân tích Địa điểm (TOPSIS)")

        # Cảnh báo đặt dưới hàng nút
        if invalid_quick or (len(edited) == 0):
            st.warning("Dữ liệu thiếu hoặc không hợp lệ.")

    if selected_scenario not in ("--- Tạo mô hình mới ---", "Tạo mô hình mới"):
        st.subheader(f"Trọng số hiện tại: **{selected_scenario}**")
        defaults = _load_default_weights()
        key_lower = str(selected_scenario).strip().lower()
        current_weights = all_weights.get(selected_scenario, {})
        if not current_weights and key_lower in ("office", "warehouse", "factory"):
            current_weights = defaults.get(key_lower, {})

        if current_weights:
            st.session_state["_default_display_model"] = selected_scenario
            st.session_state["_default_display_weights"] = current_weights
            dfw = pd.DataFrame([(nice_name(k), v) for k, v in current_weights.items()], columns=["Tiêu chí", "Trọng số"]).sort_values("Trọng số", ascending=False).reset_index(drop=True)
            dfw = add_index_col(dfw, "STT")
            display_table(dfw, bold_first_col=True, fixed_height=None)

            customize_toggle = st.toggle("Customize", value=False, key="default_customize_toggle")
            if customize_toggle:
                if "show_customization_tabs" in globals():
                    temp_dict = {selected_scenario: current_weights}
                    show_customization_tabs(temp_dict, model_name_placeholder=selected_scenario)
                else:
                    quick_customize_editor(current_weights, selected_scenario)
            else:
                if st.button("Tiếp tục qua trang phân tích", use_container_width=True):
                    st.session_state["selected_model_for_topsis"] = selected_scenario
                    st.session_state["selected_weights_for_topsis"] = current_weights
                    go("Phân tích Địa điểm (TOPSIS)")
        else:
            st.warning("Mô hình này chưa có trọng số.")
            st.info("Bật Customize để tự tạo trọng số cho kịch bản này.")
            if st.toggle("Customize", value=True, key="default_customize_toggle_empty"):
                if "show_customization_tabs" in globals():
                    show_customization_tabs({}, model_name_placeholder=selected_scenario)
                else:
                    quick_customize_editor({}, selected_scenario)
    else:
        st.info("Tạo mô hình mới.")
        if "show_customization_tabs" in globals():
            show_customization_tabs(all_weights)
        else:
            quick_customize_editor({}, "NewModel")

# =============== PAGE 4: TOPSIS ===============
elif page == "Phân tích Địa điểm (TOPSIS)":
    st.header("Trang 4: Xếp hạng Địa điểm TOPSIS")

    try:
        with open(F("weights.yaml"), 'r', encoding='utf-8') as f:
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
            data_path="AHP_Data_synced_fixed.xlsx",
            json_path="metadata.json",
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
        with open(F("weights.yaml"), 'r', encoding='utf-8') as f:
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
            df_data = pd.read_excel(F("AHP_Data_synced_fixed.xlsx"))
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

    geojson_file = "quan7_geojson.json"
    ranking_file = f"ranking_result_{model_to_map}.xlsx"

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
