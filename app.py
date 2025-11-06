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

# --- Save notification helper ---
def _notify_saved(ok: bool):
    if ok:
        try:
            st.toast("Đã lưu", icon="✅")
        except Exception:
            st.success("Đã lưu")
    else:
        st.error("Lưu thất bại.")


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

def _ui_delete_features(selected_scenario: str, current_weights: dict):
    st.markdown("**Xóa tiêu chí khỏi kịch bản**")
    features = list(current_weights.keys())
    del_opts = [f for f in features if f not in ("ward", "ward_id")]
    sel = st.multiselect("Chọn tiêu chí cần xóa", options=del_opts, key=f"del_feats_{selected_scenario}")
    if st.button("Xóa feature đã chọn", disabled=(len(sel) == 0), key=f"btn_del_feats_{selected_scenario}"):
        to_rm = set(sel)
        new_w = {k: v for k, v in current_weights.items() if k not in to_rm}
        if len(new_w) == 0:
            st.error("Không thể xóa hết tiêu chí. Cần ít nhất 1 tiêu chí.")
            return
        s = float(sum(new_w.values()))
        if s <= 0:
            # nếu tổng = 0 thì phân bổ đều
            even = 1.0 / len(new_w)
            new_w = {k: even for k in new_w}
        else:
            new_w = {k: float(v)/s for k, v in new_w.items()}
        try:
            ok, _ = save_weights_to_yaml(new_w, selected_scenario, filename=F("data/weights.yaml"))
        except Exception:
            ok = False
        _notify_saved(ok)
        # reset lưới pairwise cache để tránh lệch kích thước sau khi xóa feature
        st.session_state.pop(f"pw_edit_{selected_scenario}", None)
        st.rerun()


def _load_defaultweights_all(path="data/weights.yaml"):
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

@st.cache_data
def _load_all_weights_for_options():
    import yaml, os
    out = {}
    path = "data/weights.yaml"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
                if isinstance(d, dict):
                    out.update({str(k): d[k] for k in d.keys()})
        except Exception:
            pass
    return out  # dict tên -> weights

def _scenario_options():
    names = sorted(_load_all_weights_for_options().keys(), key=str.casefold)
    return ["Tạo mô hình mới"] + names

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

def _direct_rating_2col(features: list[str], defaults: dict[str, int], key_prefix: str) -> dict[str, int]:
    """Render chấm điểm 1–10 theo 2 cột, trả về dict feature->score."""
    cols = st.columns(2)
    scores: dict[str, int] = {}
    for i, f in enumerate(features):
        with cols[i % 2]:
            scores[f] = _block10_editor(
                label=nice_name(f),
                default=int(defaults.get(f, 5)),
                key=f"{key_prefix}_{f}",
            )
    return scores

def load_metadata():
    try:
        with open(F("metadata.json"), "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return {}

def pie_compare_weight(original_weights: dict, new_weights: dict,
                       title_left="1. Gốc", title_right="2. Mới"):
    import pandas as pd, altair as alt

    # domain cố định theo Gốc để hai pie thẳng hàng
    w0 = pd.DataFrame({
        "criterion": list(original_weights.keys()),
        "weight": [float(v) for v in original_weights.values()]
    }).sort_values("weight", ascending=False).reset_index(drop=True)
    domain = w0["criterion"].tolist()
    w0["weight"] = w0["weight"] / (w0["weight"].sum() or 1.0)
    w0["pct"] = w0["weight"] * 100

    w1 = pd.DataFrame({"criterion": domain}).merge(
        pd.DataFrame({
            "criterion": list(new_weights.keys()),
            "weight": [float(v) for v in new_weights.values()]
        }),
        on="criterion", how="left"
    ).fillna(0.0)
    w1["weight"] = w1["weight"] / (w1["weight"].sum() or 1.0)
    w1["pct"] = w1["weight"] * 100

    def _pie(df, title):
        base = alt.Chart(df).encode(
            theta=alt.Theta("weight:Q", stack=True),
            color=alt.Color("criterion:N",
                           sort=domain,
                           scale=alt.Scale(domain=domain)),
            order=alt.Order("criterion:N", sort="ascending"),
        )
        # KHÔNG dùng startAngle
        pie = base.mark_arc(innerRadius=70, outerRadius=130)

        # nhãn ngoài, ẩn lát <3% để đỡ rối
        labels = (base.transform_filter(alt.datum.pct >= 3)
                        .mark_text(radius=160, size=16, color="#e6e6e6")
                        .encode(text=alt.Text("pct:Q", format=".1f")))

        return (pie + labels).properties(
            width=420, height=420,
            title=alt.TitleParams(title, fontSize=24, fontWeight="bold",
                                  anchor="middle")
        )

    return _pie(w0, title_left) | _pie(w1, title_right)

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
                with open(F("data/weights.yaml"),"r",encoding="utf-8") as f:
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
    import os, yaml, pandas as pd, numpy as _np

    _SAATY_LABELS = ["1/9","1/8","1/7","1/6","1/5","1/4","1/3","1/2","1","2","3","4","5","6","7","8","9"]
    _SAATY_VALUES = [1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9]
    _VAL_BY_LABEL = {lab: val for lab, val in zip(_SAATY_LABELS, _SAATY_VALUES)}

    def _union_criteria_from_weights(weights_yaml_path: str = F("data/weights.yaml")) -> list[str]:
        opts = set()
        if os.path.exists(weights_yaml_path):
            try:
                with open(weights_yaml_path, "r", encoding="utf-8") as f:
                    all_w = yaml.safe_load(f) or {}
                    if isinstance(all_w, dict):
                        for _, w in all_w.items():
                            if isinstance(w, dict):
                                for k in w.keys():
                                    opts.add(str(k))
            except Exception:
                pass
        return sorted(opts)

    def _normalize_columns(A):
        A = _np.array(A, dtype=float)
        s = A.sum(axis=0); s[s==0] = 1.0
        return A / s


    def _pairwise_matrix_editor(criteria: list[str], session_key: str):
        import pandas as _pd
        n = len(criteria)

        key_df = f"{session_key}_df"  # CHỈ dùng cho widget (không ghi vào session_state[key_df])
        store_key = f"{session_key}_store"  # Dùng để lưu bản sao người dùng chỉnh

        # Khởi tạo nguồn dữ liệu cho editor từ store_key (không phải key_df)
        if store_key in st.session_state:
            df = st.session_state[store_key].copy()
            if not isinstance(df, _pd.DataFrame) or df.shape != (n, n):
                df = _pd.DataFrame(
                    [["—" if i == j else ("" if i > j else "1") for j in range(n)] for i in range(n)],
                    columns=[nice_name(c) for c in criteria],
                    index=[nice_name(c) for c in criteria],
                )
        else:
            df = _pd.DataFrame(
                [["—" if i == j else ("" if i > j else "1") for j in range(n)] for i in range(n)],
                columns=[nice_name(c) for c in criteria],
                index=[nice_name(c) for c in criteria],
            )

        col_cfg = {
            nice_name(c): st.column_config.SelectboxColumn(
                label=nice_name(c), options=_SAATY_LABELS, width="small"
            )
            for c in criteria
        }
        st.caption("Nhập trên tam giác trên. Ô '—' = 1. Tam giác dưới tự động nghịch đảo.")
        df_edit = st.data_editor(
            df, hide_index=False, use_container_width=True,
            num_rows="fixed", column_config=col_cfg, key=key_df
        )

        # LƯU vào store_key, KHÔNG ghi vào key_df
        st.session_state[store_key] = df_edit.copy()

        # Kết xuất ma trận
        A = _np.ones((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    A[i, j] = 1.0
                elif i < j:
                    lab = str(df_edit.iloc[i, j]).strip()
                    if lab in _VAL_BY_LABEL:
                        A[i, j] = float(_VAL_BY_LABEL[lab])
                        A[j, i] = 1.0 / A[i, j]
        return A

    def _block10_editor(label: str, default: int, key: str):
        try:
            val = st.segmented_control(label=label, options=list(range(1, 11)), default=int(default), format_func=str, key=key)
        except Exception:
            val = st.radio(label, list(range(1, 11)), index=int(default)-1, horizontal=True, key=key)
        v = int(val)
        filled = "".join(f'<span class="sq {"filled" if i <= v else ""}"></span>' for i in range(1, 11))
        st.markdown(f'<div class="ahp-bar">{filled}</div>', unsafe_allow_html=True)
        return v

    st.markdown(
        """
        <style>
        thead tr th div[data-testid="stMarkdownContainer"] p { white-space: nowrap; }
        .stDataFrame table { table-layout: fixed; }
        .stDataFrame [data-testid="stDataFrameResizable"] { overflow: hidden !important; }
        [data-testid="stSegmentedControl"] { max-width: 360px; }
        [data-testid="stSegmentedControl"] label { border-radius: 0 !important; min-width: 32px; height: 32px; line-height: 28px; padding: 0; text-align: center; }
        .ahp-bar { display: inline-flex; gap: 6px; margin-top: 6px; }
        .ahp-bar .sq { width: 14px; height: 14px; border: 1px solid rgba(255,255,255,0.35); }
        .ahp-bar .sq.filled { background: rgba(255, 75, 75, 0.55); border-color: rgba(255, 75, 75, 0.75); }
        </style>
        """,
        unsafe_allow_html=True
    )

    weights_file = F("data/weights.yaml")
    all_weights = {}
    if os.path.exists(weights_file):
        try:
            with open(weights_file, "r", encoding="utf-8") as f:
                all_weights = yaml.safe_load(f) or {}
        except Exception as e:
            st.error(f"Lỗi đọc weights.yaml: {e}")

    st.subheader("1. Lựa chọn Kịch bản (Scenario)")
    model_list = ["Tạo mô hình mới"] + list(all_weights.keys())

    # xác định index mặc định
    _def_idx = 0
    _after = st.session_state.pop("_after_delete_select", None)
    if _after and _after in model_list:
        _def_idx = model_list.index(_after)

    selected_scenario = st.selectbox(
        "Chọn một kịch bản có sẵn hoặc tạo mới:",
        model_list,
        index=_def_idx,
        key="scenario_selectbox",
    )

    def _protected_scenarios():
        import os, yaml
        prot = {"Office", "Default", "Baseline"}  # scenario mặc định/khóa
        # nếu có file default riêng thì gom thêm
        for cand in [F("defaultweights.yaml"), F("data/defaultweights.yaml"), F("default_weights.yaml")]:
            try:
                if os.path.exists(cand):
                    with open(cand, "r", encoding="utf-8") as f:
                        d = yaml.safe_load(f) or {}
                        if isinstance(d, dict):
                            prot.update({str(k) for k in d.keys()})
            except Exception:
                pass
        return prot

    ac1, ac2 = st.columns([2, 1])

    # Nút "Tiếp tục tới phân tích"
    with ac1:
        disabled_next = (selected_scenario == "Tạo mô hình mới")
        st.button(
            "Tiếp tục tới phân tích (TOPSIS)",
            use_container_width=True,
            disabled=disabled_next,
            key="btn_next_to_topsis",
            on_click=lambda: (
                st.session_state.setdefault("selected_model_for_topsis", selected_scenario),
                st.session_state.setdefault("auto_run_topsis", True),
                st.session_state.setdefault("customize_mode", False),
                st.session_state.update(
                    {"selected_model_for_topsis": selected_scenario, "auto_run_topsis": True, "customize_mode": False}),
                go("Phân tích Địa điểm (TOPSIS)")
            )
        )
        if disabled_next:
            st.caption("Chọn một kịch bản trước khi tiếp tục.")

    # Nút "Xóa kịch bản" – chỉ cho phép xóa kịch bản do user tạo
    with ac2:
        prot = _protected_scenarios()
        can_delete = (selected_scenario not in ("Tạo mô hình mới",)) and (selected_scenario not in prot)
        if st.button("Xóa kịch bản", use_container_width=True, disabled=not can_delete, key="btn_delete_scenario"):
            try:
                import yaml, os

                wf = F("data/weights.yaml")
                data = {}
                if os.path.exists(wf):
                    with open(wf, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                if isinstance(data, dict) and selected_scenario in data:
                    data.pop(selected_scenario, None)
                    with open(wf, "w", encoding="utf-8") as f:
                        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=True)
                    st.success(f"Đã xóa kịch bản '{selected_scenario}'.")
                    st.session_state["_after_delete_select"] = "Tạo mô hình mới"
                    st.rerun()
                else:
                    st.warning("Không tìm thấy kịch bản để xóa.")
            except Exception as e:
                st.error(f"Lỗi khi xóa: {e}")
        if not can_delete:
            st.caption("Không thể xóa kịch bản mặc định hoặc khi chưa chọn kịch bản.")

    if 'ahp_wizard' not in st.session_state:
        st.session_state.ahp_wizard = {"step": 1, "name": "", "selected": []}

    if selected_scenario == "Tạo mô hình mới":
        w = st.session_state.ahp_wizard
        if w["step"] == 1:
            st.subheader("Bước 1: Chọn tiêu chí")
            available = _union_criteria_from_weights()
            ui_cols = st.columns([1,1,4])
            with ui_cols[0]:
                if st.button("Chọn tất cả"):
                    for c in available: st.session_state[f"chk_{c}"] = True
            with ui_cols[1]:
                if st.button("Bỏ chọn tất cả"):
                    for c in available: st.session_state[f"chk_{c}"] = False
            cols = st.columns(3)
            chosen = set(w.get("selected") or [])
            for idx, c in enumerate(available):
                with cols[idx%3]:
                    chk = st.checkbox(nice_name(c), value=(c in chosen), key=f"chk_{c}")
                    if chk: chosen.add(c)
                    else: chosen.discard(c)
            model_name = st.text_input("Đặt tên mô hình", value=w.get("name") or "Custom")
            st.session_state.ahp_wizard["name"] = model_name
            st.session_state.ahp_wizard["selected"] = sorted(list(chosen))
            if st.button("Next", disabled=(len(chosen)<2 or not model_name.strip())):
                st.session_state.ahp_wizard["step"] = 2; st.rerun()
        else:
            selected = st.session_state.ahp_wizard["selected"]
            model_name = st.session_state.ahp_wizard["name"].strip()
            st.subheader("Bước 2: So sánh/Chấm điểm")
            mode_new = st.radio("Chọn phương thức nhập", ["Pairwise (AHP)", "Direct rating (1–10)"], horizontal=True, key="new_model_mode")
            if mode_new == "Pairwise (AHP)":
                A_input = _pairwise_matrix_editor(selected, session_key="pw_new_editor")
                M_norm = _normalize_columns(A_input)
                if st.button("Tính AHP và Lưu"):
                    weights_vec, cr = calculate_ahp_weights(M_norm)
                    if weights_vec is not None:
                        s = float(sum(weights_vec)) or 1.0
                        weights_norm = {k: float(v)/s for k, v in zip(selected, weights_vec)}
                        ok, saved_name = save_weights_to_yaml(weights_norm, model_name, filename=F("data/weights.yaml"))
                        if ok:
                            _notify_saved(True)
                        else:
                            st.error("Lưu thất bại.")
                    else:
                        st.error("Không tính được trọng số.")
            else:
                st.caption("Direct rating 1–10. Hệ thống chuẩn hoá về tổng = 1.")
                scores = _direct_rating_2col(selected, {}, "new_block")
                s = sum(scores.values()) or 1
                weights_norm = {k: float(v)/s for k, v in scores.items()}
                dfw = pd.DataFrame([(nice_name(k), scores[k], weights_norm[k]) for k in selected],
                                   columns=["Tiêu chí","Điểm 1–10","Trọng số (chuẩn hoá)"])
                st.dataframe(dfw, hide_index=True, use_container_width=True)
                if st.button("Lưu mô hình (Direct rating)"):
                    ok, saved_name = save_weights_to_yaml(weights_norm, model_name, filename=F("data/weights.yaml"))
                    _notify_saved(ok)
            if st.button("Quay lại bước 1"):
                st.session_state.ahp_wizard["step"] = 1; st.rerun()
    else:
        st.subheader(f"Trọng số hiện tại: **{selected_scenario}**")
        current_weights = all_weights.get(selected_scenario, {})
        if current_weights:
            dfw = pd.DataFrame([(nice_name(k), v) for k, v in current_weights.items()],
                               columns=["Tiêu chí","Trọng số"]).sort_values("Trọng số", ascending=False).reset_index(drop=True) 
            st.dataframe(dfw, hide_index=True, use_container_width=True)
            st.markdown("---"); st.markdown("#### Customize")
            _cust_visible_key = f"cust_visible_{selected_scenario}"
            cust_visible = st.toggle("Bật Customize", value=False, key=_cust_visible_key)
            if not cust_visible:
                st.caption("Customize đang tắt.")
                st.stop()
            _ui_delete_features(selected_scenario, current_weights)
            mode = st.radio("Phương thức chỉnh", ["Pairwise (AHP)", "Direct rating (1–10)"], horizontal=True, key=f"cust_mode_{selected_scenario}")
            features = list(current_weights.keys())
            if mode == "Pairwise (AHP)":
                A_input = _pairwise_matrix_editor(features, session_key=f"pw_edit_{selected_scenario}")
                M_norm = _normalize_columns(A_input)
                weights_vec, cr = calculate_ahp_weights(M_norm)
                if weights_vec is not None:
                    s = float(sum(weights_vec)) or 1.0
                    weights_norm = {k: float(v)/s for k, v in zip(features, weights_vec)}
                    dfp = pd.DataFrame([(nice_name(k), weights_norm[k]) for k in features],
                                       columns=["Tiêu chí","Trọng số"]).sort_values("Trọng số", ascending=False).reset_index(drop=True)
                    st.dataframe(dfp, hide_index=True, use_container_width=True)
                if st.button("Lưu thay đổi", key=f"save_pw_{selected_scenario}"):
                    ok, saved_name = save_weights_to_yaml(weights_norm, selected_scenario, filename=F("data/weights.yaml"))
                    _notify_saved(ok)
            else:
                init = {f: max(1, min(10, int(round(float(current_weights.get(f,0))*10)))) for f in features}
                scores = _direct_rating_2col(features, init, f"cust_block_{selected_scenario}")
                s = sum(scores.values()) or 1
                new_w = {f: float(v)/s for f, v in scores.items()}
                dfp = pd.DataFrame([(nice_name(k), scores[k], new_w[k]) for k in features],
                                   columns=["Tiêu chí","Điểm 1–10","Trọng số (chuẩn hoá)"])
                st.dataframe(dfp, hide_index=True, use_container_width=True)
                if st.button("Lưu thay đổi", key=f"save_num_{selected_scenario}"):
                    ok, saved_name = save_weights_to_yaml(new_w, selected_scenario, filename=F("data/weights.yaml"))
                    _notify_saved(ok)

elif page == "Phân tích Địa điểm (TOPSIS)":
    st.header("Trang 4: Xếp hạng Địa điểm TOPSIS")

    try:
        with open(F("data/weights.yaml"), 'r', encoding='utf-8') as f:
            all_weights = yaml.safe_load(f) or {}
            model_names = list(all_weights.keys())
            if not model_names:
                st.warning("Chưa có mô hình AHP.")
                st.stop()
    except FileNotFoundError:
        st.error("Thiếu 'data/weights.yaml'.")
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
        # === Weights summary (table + pie) ===
        weights_dict = all_weights.get(model_name, {}) or {}
        if isinstance(weights_dict, dict) and weights_dict:
            wdf = pd.DataFrame(
                [(nice_name(k), round(float(v)*10, 2), round(float(v)*100, 2)) for k, v in weights_dict.items()],
                columns=["Tiêu chí", "Điểm (1–10)", "Trọng số (%)"]
            ).sort_values("Trọng số (%)", ascending=False).reset_index(drop=True)
            c1, c2 = st.columns([1,1])
            with c1:
                st.subheader("Bảng tiêu chí")
                display_table(wdf, bold_first_col=True, fixed_height=280)
            with c2:
                st.subheader("Phân bổ trọng số")
                try:
                    # wdf: DataFrame có cột "Tiêu chí" và "Trọng số (%)"
                    wdf = wdf.sort_values("Trọng số (%)", ascending=False).reset_index(drop=True)
                    wdf["_label"] = wdf["Trọng số (%)"].round().astype(int).astype(str) + "%"

                    LABEL_MIN = 0

                    base = alt.Chart(wdf).encode(
                        theta=alt.Theta("Trọng số (%)", stack=True),
                        color=alt.Color("Tiêu chí", sort=wdf["Tiêu chí"].tolist()),
                        order=alt.Order("Trọng số (%)", sort="descending")
                    )

                    pie = base.mark_arc(outerRadius=120, innerRadius=55)

                    text = (
                        base.transform_filter(alt.datum["Trọng số (%)"] >= LABEL_MIN)
                        .mark_text(radius=148)  # nhãn nằm ngoài
                        .encode(text=alt.Text("_label:N"))
                    )

                    st.altair_chart(pie + text, use_container_width=True)
                except Exception:
                    pass
            st.divider()
        # === Run TOPSIS ===
        report_df = run_topsis_model(
            data_path=F("AHP_Data_synced_fixed.xlsx"),
            json_path=F("metadata.json"),
            analysis_type=model_name,
            all_criteria_weights=all_weights
        )
        if report_df is not None:
            st.session_state['last_topsis_df'] = report_df.copy()
            st.subheader("Xếp hạng đầy đủ")
            show = report_df.copy()
            show = add_index_col(show, "STT")
            display_table(show, bold_first_col=True, fixed_height=380)
            st.divider()
            cols = st.columns(4)
            with cols[0]:
                if st.button("Rerun DSS", use_container_width=True):
                    st.rerun()
            with cols[1]:
                st.button("Map View", on_click=switch_to_map_view, use_container_width=True)
            with cols[2]:
                st.button("Sensitivity", on_click=switch_to_sensitivity, use_container_width=True)
            with cols[3]:
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
        with open(F("data/weights.yaml"), 'r', encoding='utf-8') as f:
            all_weights = yaml.safe_load(f) or {}
            model_names = list(all_weights.keys())
            if not model_names:
                st.warning("Chưa có mô hình AHP.")
                st.stop()
    except FileNotFoundError:
        st.error("Thiếu 'data/weights.yaml'.")
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
            original_df, new_df = run_what_if_analysis(selected_model, normalized_weights, all_weights)
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

                chart = pie_compare_weight(original_weights, normalized_weights,
                                           title_left="1. Gốc", title_right="2. Mới")
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

    hue_label = st.radio("Màu heatmap", ["Xanh lá", "Đỏ"], horizontal=True, key="map_hue")
    hue = "green" if hue_label == "Xanh lá" else "red"

    geojson_file = "quan7_geojson.json"
    ranking_file = f"ranking_result_{model_to_map}.xlsx"

    with open(F(geojson_file), 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    df_ranking = pd.read_excel(F(ranking_file))

    ranking_lookup = {
        str(row['Tên phường']).replace(" ", ""): row.to_dict()
        for _, row in df_ranking.iterrows()
    }

    def mono_palette(h):
        if h == "red":
            return [
                [255,235,230,220],[255,210,195,220],[255,184,160,220],
                [248,150,120,220],[232,112,84,220],[206,72,58,220],[170,40,40,220],
            ]
        return [
            [230,248,235,220],[201,236,214,220],[166,222,189,220],
            [120,206,160,220],[72,188,128,220],[34,163,98,220],[16,128,74,220],
        ]
    PAL = mono_palette(hue); BINS = len(PAL)

    def color_from_score_discrete(score: float):
        t = 0.0 if score is None else max(0.0, min(1.0, float(score)))
        idx = min(BINS - 1, int(round(t * (BINS - 1))))
        return PAL[idx]

    # Top 1–3 dùng 3 mức đậm nhất cùng tông, viền vẫn mỏng như các ô khác
    def color_for_rank(rank, score):
        try: rnk = int(rank)
        except: rnk = 0
        if rnk == 1: return PAL[-1]
        if rnk == 2: return PAL[-2]
        if rnk == 3: return PAL[-3]
        return color_from_score_discrete(score)

    # Gắn thuộc tính
    for feature in geojson_data.get('features', []):
        name_raw = feature.get('properties', {}).get('name')
        if not name_raw: continue
        key = str(name_raw).replace(" ", "")
        info = ranking_lookup.get(key)

        if info is not None:
            rank = int(info.get('Rank'))
            try: score = float(info.get('Điểm TOPSIS (0-1)'))
            except: score = 0.0
            feature['properties']['Rank'] = rank
            feature['properties']['Score'] = score
            feature['properties']['color'] = color_for_rank(rank, score)
        else:
            feature['properties']['Rank'] = "N/A"
            feature['properties']['Score'] = None
            feature['properties']['color'] = PAL[0]

    st.subheader("Bản đồ Xếp hạng TOPSIS")
    st.caption("Một tông màu. Top 1–3 đậm hơn, mọi viền đều mỏng.")

    view_state = pdk.ViewState(latitude=10.73, longitude=106.72, zoom=13, pitch=0, bearing=0)
    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson_data,
        opacity=0.9,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="properties.color",
        get_line_color=[25, 25, 25, 160],
        get_line_width=60,          # viền mỏng cho tất cả
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        "html": "<b>Phường:</b> {name}<br/><b>Hạng:</b> {Rank}<br/><b>Điểm TOPSIS:</b> {Score}",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }
    deck = pdk.Deck(layers=[layer], initial_view_state=view_state,
                    map_style=pdk.map_styles.LIGHT, tooltip=tooltip)
    st.pydeck_chart(deck, use_container_width=True)
