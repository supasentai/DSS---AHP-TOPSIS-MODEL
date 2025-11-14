# app.py — cleaned & organized
import os
import re
import json
import yaml
import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from html import escape as _esc
from pathlib import Path


# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

def F(name: str) -> str:
    """Absolute path for reading under ./data (fallback to root)."""
    p = DATA_DIR / name
    q = ROOT / name
    return str(p if p.exists() else q)

def FW(name: str) -> str:
    """Absolute path for writing under ./data; create parents."""
    p = DATA_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


# =========================
# External modules
# =========================
try:
    from ahp_module import calculate_ahp_weights, save_weights_to_yaml
    from topsis_module import run_topsis_model
    from sensitivity_module import run_what_if_analysis
    from report_module import create_full_report
except ImportError as e:
    st.error(
        f"Lỗi import module: {e}. Kiểm tra các file ahp_module.py, topsis_module.py, sensitivity_module.py."
    )
    st.stop()


# =========================
# Streamlit page config
# =========================
st.set_page_config(page_title="DSS Địa điểm Việt Nam", page_icon="🦈", layout="wide")

st.markdown("""
<style>
.model-status{font-size:2rem;font-weight:600;margin:.25rem 0 0;color:#f59e0b}
</style>
""", unsafe_allow_html=True)

# =========================
# UI helpers
# =========================
def _notify_saved(ok: bool):
    try:
        (st.toast if ok else st.error)("Đã lưu" if ok else "Lưu thất bại.", icon="✅" if ok else None)
    except Exception:
        (st.success if ok else st.error)("Đã lưu" if ok else "Lưu thất bại.")

def inject_global_css():
    st.markdown(
        """
<style>
.styled-table{width:100%;border-collapse:collapse;border-spacing:0;table-layout:auto;margin-bottom:24px}
.styled-table th,.styled-table td{padding:12px 14px;text-align:center;vertical-align:middle}
.styled-table th{white-space:nowrap}
.fixed-height{max-height:420px;overflow:auto;margin-bottom:32px}

/* light */
.styled-table{border:4px solid #1F2937;background:#FFF}
.styled-table thead th{font-weight:800;background:#F1F5F9;color:#0F172A;border:4px solid #1F2937}
.styled-table tbody td{background:#FFF;color:#0F172A;border:4px solid #1F2937}
.styled-table tbody td:first-child{background:#F1F5F9;color:#0F172A}

/* dark */
@media (prefers-color-scheme: dark){
  .styled-table{border:4px solid #94A3B8;background:#0B1220}
  .styled-table thead th{background:#0E1A2B;color:#F8FAFC;border:4px solid #94A3B8}
  .styled-table tbody td{background:#0B1220;color:#E5E7EB;border:4px solid #94A3B8}
  .styled-table tbody td:first-child{background:#0E1A2B;color:#F8FAFC}
}

/* header-only tooltips */
/* tooltip sizing override: do not shrink to cell width */
.styled-table th[data-tip]{position:relative; overflow: visible;}
.styled-table th[data-tip]:hover::after{
  white-space: nowrap;
  width: max-content;
  max-width: none;
  z-index: 9999;
}
.styled-table th[data-tip]:hover::before{ z-index: 9999; }

.styled-table th[data-tip]{position:relative}
.styled-table th[data-tip]:hover::before{
  content:"";position:absolute;left:50%;top:calc(100% + 2px);transform:translateX(-50%);
  border:6px solid transparent;border-bottom-color: rgba(15,15,20,.98)
}
.styled-table th[data-tip]:hover::after{
  content: attr(data-tip);position:absolute;left:50%;top:calc(100% + 14px);transform:translateX(-50%);
  background: rgba(15,15,20,.98);color:#fff;padding:12px 14px;border-radius:10px;border:1px solid rgba(255,255,255,.12)
}
.styled-table td[data-tip],.styled-table td [data-tip]{position:static}
.styled-table td[data-tip]::before,.styled-table td[data-tip]::after,
.styled-table td [data-tip]::before,.styled-table td [data-tip]::after{content:none}
</style>
        """,
        unsafe_allow_html=True,
    )

def nice_name(col: str) -> str:
    return str(col).replace("_", " ").strip().title()


# Chuẩn hóa tên tỉnh/thành để join Excel <-> GeoJSON (không phụ thuộc thư viện ngoài)
# Đầu tiên khai báo mapping chữ cái (thường) → không dấu, sau đó tự sinh thêm bản chữ hoa.
_VIET_BASE = {
    # a
    "à": "a","á": "a","ả": "a","ã": "a","ạ": "a",
    "ă": "a","ằ": "a","ắ": "a","ẳ": "a","ẵ": "a","ặ": "a",
    "â": "a","ầ": "a","ấ": "a","ẩ": "a","ẫ": "a","ậ": "a",
    # e
    "è": "e","é": "e","ẻ": "e","ẽ": "e","ẹ": "e",
    "ê": "e","ề": "e","ế": "e","ể": "e","ễ": "e","ệ": "e",
    # i
    "ì": "i","í": "i","ỉ": "i","ĩ": "i","ị": "i",
    # o
    "ò": "o","ó": "o","ỏ": "o","õ": "o","ọ": "o",
    "ô": "o","ồ": "o","ố": "o","ổ": "o","ỗ": "o","ộ": "o",
    "ơ": "o","ờ": "o","ớ": "o","ở": "o","ỡ": "o","ợ": "o",
    # u
    "ù": "u","ú": "u","ủ": "u","ũ": "u","ụ": "u",
    "ư": "u","ừ": "u","ứ": "u","ử": "u","ữ": "u","ự": "u",
    # y
    "ỳ": "y","ý": "y","ỷ": "y","ỹ": "y","ỵ": "y",
    # d
    "đ": "d",
}
_char_map = {}
for ch, rep in _VIET_BASE.items():
    _char_map[ch] = rep
    _char_map[ch.upper()] = rep.upper()
_VIET_TRANS = str.maketrans(_char_map)


def norm_province_name(s: str) -> str:
    """Đưa tên tỉnh/thành về dạng chuẩn ASCII để join giữa Excel và GeoJSON."""
    import re as _re
    s = str(s).strip().lower()
    # bỏ từ loại hình hành chính
    for w in ("tinh", "tỉnh", "thanh pho", "thành phố", "tp.", "tp ", "tp"):
        s = s.replace(w, "")
    # bỏ dấu tiếng Việt → ascii
    s = s.translate(_VIET_TRANS)
    # bỏ khoảng trắng và ký tự không chữ-số
    s = _re.sub(r"[^a-z0-9]+", "", s)
    # ánh xạ các viết tắt đặc biệt
    special = {
        "brvt": "bariavungtau",
        "hochiminhcity": "hochiminh",
        "tphochiminh": "hochiminh",
    }
    return special.get(s, s)


def _next_clone_name(base_name, existing):
    base = str(base_name).strip() or "Custom"
    exist = {str(x).strip().lower() for x in existing}
    i = 1
    cand = f"{base}_{i}"
    while cand.strip().lower() in exist:
        i += 1
        cand = f"{base}_{i}"
    return cand

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
        return re.sub(r'\s+', ' ', s).strip().lower()

    tips = {_norm(k): v for k, v in header_tooltips.items()}
    new_cells = []
    for thm in ths:
        cell = thm.group(0)
        label = _norm(thm.group(1))
        tip = tips.get(label)
        if tip and 'data-tip=' not in cell:
            esc = _esc(str(tip), quote=True)
            cell = re.sub(r'(<th\b[^>]*)>', f'\\1 data-tip="{esc}">', cell, flags=re.I)
        new_cells.append(cell)
    new_head = ''.join(new_cells)
    return html_table[:m.start(1)] + new_head + html_table[m.end(1):]

def apply_display_names(df: pd.DataFrame, name_map: dict | None = None) -> pd.DataFrame:
    out = df.rename(columns=name_map or {})
    out.columns = [nice_name(c) for c in out.columns]
    return out

def add_index_col(df: pd.DataFrame, label: str = "STT") -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out.insert(0, label, range(1, len(out) + 1))
    return out

def to_html_table(df: pd.DataFrame, bold_first_col: bool = True) -> str:
    df2 = df.copy()
    drop_candidates = [
        c for c in df2.columns
        if str(c).strip() == "ward_id"
        or str(c).strip().lower().replace(" ", "").replace("-", "") in {"wardid", "maphuong", "maward"}
    ]
    # also drop any pandas index dump columns like 'Unnamed: 0'
    drop_unnamed = [c for c in df2.columns if str(c).strip().lower().startswith('unnamed')]
    drop_candidates = list(drop_candidates) + drop_unnamed
    if drop_candidates:
        df2 = df2.drop(columns=drop_candidates, errors="ignore")
    df2.columns = [nice_name(c) if "_" in str(c) else str(c) for c in df2.columns]
    if bold_first_col and df2.shape[1] > 0:
        first = df2.columns[0]
        df2[first] = df2[first].map(lambda x: f"<strong>{x}</strong>")
    return df2.to_html(index=False, escape=False, classes="styled-table")


def display_table(
    df,
    bold_first_col: bool = True,
    fixed_height: int | None = 420,
    header_tooltips=None,
    zoomable: bool | None = None,
    zoom_key: str | None = None,
):
    """
    Hiển thị bảng:
    - Bảng lớn (nhiều cột) -> dùng st.dataframe (có nút phóng to / fullscreen).
    - Bảng nhỏ -> dùng HTML .styled-table cố định, không méo layout.
    """
    # Auto phát hiện bảng lớn
    if zoomable is None:
        try:
            n_cols = df.shape[1]
        except Exception:
            n_cols = 0
        zoomable = n_cols >= 8

    # Bảng lớn: dùng st.dataframe để có nút "phóng to" và scroll tốt hơn
    # và canh giữa (center) tất cả cell + header thông qua pandas Styler.
    if zoomable:
        h = fixed_height or 420
        try:
            _df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        except Exception:
            _df = pd.DataFrame(df)
        styler = (
            _df.style
            .set_properties(**{"text-align": "center"})
            .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
        )
        st.dataframe(styler, height=int(h), use_container_width=True)
        return

    # Bảng nhỏ: giữ style HTML cũ, fixed layout
    html_tbl = to_html_table(df, bold_first_col=bold_first_col)
    if '<table' in html_tbl:
        open_tag = html_tbl.split('>', 1)[0]
        if 'class=' not in open_tag:
            html_tbl = html_tbl.replace('<table', '<table class="styled-table"', 1)
        elif 'styled-table' not in open_tag:
            html_tbl = html_tbl.replace('class="', 'class="styled-table ', 1)

    # Xoá data-tip cũ và inject lại nếu có
    html_tbl = re.sub(r'\sdata-tip="[^"]*"', '', html_tbl)
    if header_tooltips:
        html_tbl = _inject_tooltips_on_th(html_tbl, header_tooltips)

    style_parts = []
    if fixed_height is not None:
        style_parts.append(f"max-height:{int(fixed_height)}px")
        style_parts.append("overflow:auto")

    h_style = ";".join(style_parts) + (";" if style_parts else "")
    st.markdown(f'<div class="fixed-height" style="{h_style}">{html_tbl}</div>', unsafe_allow_html=True)


@st.cache_data
def _load_all_weights_for_options():
    out = {}
    path = F("data/weights.yaml")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
            if isinstance(d, dict):
                out.update({str(k): d[k] for k in d.keys()})
    return out

def _scenario_options():
    names = sorted(_load_all_weights_for_options().keys(), key=str.casefold)
    return ["Tạo mô hình mới"] + names

def _direct_rating_2col(features: list[str], defaults: dict[str, int], key_prefix: str) -> dict[str, int]:
    cols = st.columns(2)
    scores: dict[str, int] = {}
    for i, f in enumerate(features):
        with cols[i % 2]:
            scores[f] = _block10_editor(label=nice_name(f), default=int(defaults.get(f, 5)), key=f"{key_prefix}_{f}")
    return scores

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
        out[c] = f"{dn} ({tp.title()})" if tp else dn
    return out

def summarize_weights(weights: dict | None):
    if not weights:
        return None
    total = sum(weights.values()) or 1.0
    norm = {k: v / total for k, v in weights.items()}
    top = sorted(norm.items(), key=lambda x: x[1], reverse=True)[:5]
    return {"count": len(norm), "top": top}

def pie_compare_weight(original_weights: dict, new_weights: dict,
                       title_left="1. Gốc", title_right="2. Mới"):
    w0 = pd.DataFrame(
        {"criterion": list(original_weights.keys()),
         "weight": [float(v) for v in original_weights.values()]}
    ).sort_values("weight", ascending=False).reset_index(drop=True)
    domain = w0["criterion"].tolist()
    w0["weight"] = w0["weight"] / (w0["weight"].sum() or 1.0)
    w0["pct"] = w0["weight"] * 100

    w1 = pd.DataFrame({"criterion": domain}).merge(
        pd.DataFrame({"criterion": list(new_weights.keys()),
                      "weight": [float(v) for v in new_weights.values()]}),
        on="criterion", how="left"
    ).fillna(0.0)
    w1["weight"] = w1["weight"] / (w1["weight"].sum() or 1.0)
    w1["pct"] = w1["weight"] * 100

    def _pie(df, title):
        base = alt.Chart(df).encode(
            theta=alt.Theta("weight:Q", stack=True),
            color=alt.Color("criterion:N", sort=domain, scale=alt.Scale(domain=domain)),
            order=alt.Order("criterion:N", sort="ascending"),
        )
        pie = base.mark_arc(innerRadius=70, outerRadius=130)
        labels = (base.transform_filter(alt.datum.pct >= 3)
                        .mark_text(radius=160, size=16, color="#e6e6e6")
                        .encode(text=alt.Text("pct:Q", format=".1f")))
        return (pie + labels).properties(
            width=420, height=420,
            title=alt.TitleParams(title, fontSize=24, fontWeight="bold", anchor="middle")
        )

    return _pie(w0, title_left) | _pie(w1, title_right)

def show_home_summary():
    st.subheader("Tóm tắt kết quả")
    colA, colB = st.columns([2, 3])

    with colA:
        try:
            df = pd.read_excel(F("AHP_Data_synced_fixed.xlsx"))
            metadata = load_metadata()
            id_col = "Tỉnh/Thành phố"
            non_crit_cols = (id_col, "Vùng")
            n_prov = int(df[id_col].nunique()) if id_col in df.columns else len(df)
            crits = [c for c in df.columns if c not in non_crit_cols]
            types = [metadata.get(c, {}).get("type", "") for c in crits]
            n_benefit = sum(1 for t in types if t == "benefit")
            n_cost = sum(1 for t in types if t == "cost")
            st.metric("Số Tỉnh/Thành", n_prov)
            st.metric("Số tiêu chí", len(crits), help=f"Benefit: {n_benefit} · Cost: {n_cost}")
        except Exception:
            st.info("Chưa có dữ liệu để tóm tắt.")

    with colB:
        last_model = st.session_state.get("last_saved_model") or                      st.session_state.get("topsis_model_selector") or                      st.session_state.get("whatif_model_selector")
        last_weights = st.session_state.get("last_saved_weights")
        if not last_weights and last_model:
            try:
                with open(F("data/weights.yaml"), "r", encoding="utf-8") as f:
                    yw = yaml.safe_load(f) or {}
                last_weights = yw.get(last_model)
            except Exception:
                last_weights = None
        st.markdown("**AHP gần nhất**")
        if last_model and last_weights:
            st.caption(last_model)
            summary = summarize_weights(last_weights)
            if summary:
                top_items = [(nice_name(k), v) for k, v in summary["top"]]
                dfw = pd.DataFrame(top_items, columns=["Tiêu chí", "Trọng số"]).reset_index(drop=True)
                dfw = add_index_col(dfw, "STT")
                display_table(dfw, bold_first_col=True, fixed_height=220)
        else:
            st.caption("Chưa có mô hình/ trọng số.")

    st.divider()
    st.markdown("**Kết quả TOPSIS gần nhất**")
    last_topsis_df = st.session_state.get("last_topsis_df")
    last_topsis_model = st.session_state.get("last_topsis_model")
    if last_topsis_df is not None and not last_topsis_df.empty:
        top3 = add_index_col(last_topsis_df.head(3).copy(), "STT")
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
            display_table(
                add_index_col(
                    improved[["Tên Tỉnh/Thành", "Hạng Mới", "Hạng Gốc", "Thay đổi"]].reset_index(drop=True),
                    "STT"
                ),
                True,
                200,
            )
        with c2:
            st.caption("Giảm hạng nhiều nhất")
            display_table(
                add_index_col(
                    declined[["Tên Tỉnh/Thành", "Hạng Mới", "Hạng Gốc", "Thay đổi"]].reset_index(drop=True),
                    "STT"
                ),
                True,
                200,
            )
    else:
        st.caption("Chưa chạy What-if.")


# =========================
# Session + navigation
# =========================
inject_global_css()

for k, v in {
    'criteria_names': [],
    'ahp_matrices': {},
    'customize_mode': False,
    'selected_model_for_topsis': None,
    'auto_run_topsis': False,
    'last_saved_model': None,
    'last_saved_weights': None,
    'model_for_next_page': None,
    'pending_nav': None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _restore_page_state(page_name: str):
    store = st.session_state.get("_page_states", {}).get(page_name, {})
    for k, v in store.items():
        if _is_forbidden_widget_key(k):
            continue
        if k not in st.session_state:
            st.session_state[k] = v


def _is_forbidden_widget_key(k: str) -> bool:
    s = str(k)
    # data_editor widgets created in AHP editors use keys like pw_*_df, cust_*_df
    if s.endswith("_df") and (s.startswith("pw_") or s.startswith("cust_") or "editor" in s):
        return True
    return False
def _remember_page_state(page_name: str, prefixes: list[str] | tuple[str, ...]):
    store = {}
    # include exact matches and prefix matches
    for k, v in st.session_state.items():
        if any(k == pf or k.startswith(pf) for pf in prefixes):
            store[k] = v
    all_states = dict(st.session_state.get("_page_states", {}))
    all_states[page_name] = store
    st.session_state["_page_states"] = all_states
def _remember_for(page_name: str):
    mapping = {
        "Dashboard": ['page_navigator'],
        "Tổng quan Dữ liệu": ["data_overview_"],
        "Tùy chỉnh Trọng số (AHP)": ["ahp_wizard","scenario_selectbox","new_model_mode","chk_","pw_new_editor","pw_edit_","new_block","cust_","del_feats_","save_pw_","save_num_","cust_mode_"],
        "Phân tích Địa điểm (TOPSIS)": ["scenario_selectbox","topsis_model_selector","last_topsis_df","last_topsis_model"],
        "Phân tích Độ nhạy (What-if)": ["whatif_model_selector","slider_","last_whatif_rank_changes"],
        "Map View": ["map_hue","model_for_next_page"],
    }
    _remember_page_state(page_name, mapping.get(page_name, []))

def go(page_name: str):
    st.session_state.pending_nav = page_name
    st.rerun()

def switch_to_topsis_page_and_run():
    m = st.session_state.scenario_selectbox
    st.session_state.update({
        "selected_model_for_topsis": m,
        "customize_mode": False,
        "auto_run_topsis": True,
        "last_saved_model": None,
        "last_saved_weights": None,
    })
    go("Phân tích Địa điểm (TOPSIS)")

def switch_to_map_view():
    # ưu tiên model đang chọn ở TOPSIS selectbox
    m = st.session_state.get("scenario_selectbox") \
        or st.session_state.get("topsis_model_selector") \
        or st.session_state.get("last_topsis_model")
    if m:
        st.session_state.model_for_next_page = m
    go("Map View")

def switch_to_sensitivity():
    last = st.session_state.get('last_topsis_model')
    if last:
        st.session_state["model_for_whatif"] = last
    st.session_state["page"] = "Phân tích độ nhạy (What-if)"
    st.rerun()

def switch_to_ahp_customize():
    if st.session_state.page_navigator == "Phân tích Địa điểm (TOPSIS)":
        st.session_state.scenario_selectbox = st.session_state.topsis_model_selector
    elif st.session_state.page_navigator == "Phân tích Độ nhạy (What-if)":
        st.session_state.scenario_selectbox = st.session_state.whatif_model_selector
    st.session_state.customize_mode = True
    go("Tùy chỉnh Trọng số (AHP)")

def _run_topsis_from(model_name: str):
    # handoff + autorun
    st.session_state["model_for_next_page"] = model_name
    st.session_state["topsis_autorun"] = True
    # điều hướng qua radio sidebar
    go("Phân tích Địa điểm (TOPSIS)")

# =========================
# Sidebar nav
# =========================
st.title("🦈 Hệ thống Hỗ trợ Quyết định Chọn địa điểm tại Việt Nam")
if st.session_state.pending_nav:
    st.session_state.page_navigator = st.session_state.pending_nav
    st.session_state.pending_nav = None

st.sidebar.title("Menu")

# remember current page's state before switching (captures latest widget values)
_prev_page_for_remember = st.session_state.get("page_navigator")
page = st.sidebar.radio(
    "Chọn một trang:",
    ["Dashboard", "Tổng quan Dữ liệu", "Tùy chỉnh Trọng số (AHP)", "Phân tích Địa điểm (TOPSIS)", "Phân tích Độ nhạy (What-if)", "Map View"],
    key="page_navigator",
)
_restore_page_state(page)

if _prev_page_for_remember and _prev_page_for_remember != page:
    _remember_for(_prev_page_for_remember)

# =========================
# Page: Homepage
# =========================
if page == "Dashboard":
    st.header("Dashboard tổng quan")
    st.markdown("Tổng hợp nhanh dữ liệu, mô hình AHP, kết quả TOPSIS và kịch bản What-if. Dùng các nút dưới để đi tới từng phân hệ chi tiết.")
    c1, c2, c3 = st.columns(3)
    if c1.button("Tổng quan Dữ liệu", use_container_width=True): go("Tổng quan Dữ liệu")
    if c2.button("AHP – Trọng số", use_container_width=True): go("Tùy chỉnh Trọng số (AHP)")
    if c3.button("TOPSIS – Xếp hạng", use_container_width=True): go("Phân tích Địa điểm (TOPSIS)")
    d1, d2 = st.columns(2)
    if d1.button("What-if – Độ nhạy", use_container_width=True): go("Phân tích Độ nhạy (What-if)")
    if d2.button("Map View – Bản đồ", use_container_width=True): go("Map View")

    st.divider()
    show_home_summary()



    _remember_page_state("Dashboard", ['page_navigator'])
# =========================
# Page: Data Overview
# =========================
elif page == "Tổng quan Dữ liệu":
    st.header("Trang 2: Khám phá và Tổng quan Dữ liệu")
    try:
        df = pd.read_excel(F("AHP_Data_synced_fixed.xlsx"))
        metadata = load_metadata()
    except FileNotFoundError:
        st.error("Thiếu AHP_Data_synced_fixed.xlsx hoặc metadata.json.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        st.stop()

    id_col = "Tỉnh/Thành phố"
    non_crit_cols = (id_col, "Vùng")

    name_map = {
        c: metadata.get(c, {}).get("display_name", nice_name(c))
        for c in df.columns if c not in non_crit_cols
    }

    tab1, tab2 = st.tabs(["📊 Thống kê Chung", "📈 Phân tích Từng Tiêu chí"])

    def _resolve_desc_tooltips(ddf):
        base = {
            "count": "Số bản ghi hợp lệ.",
            "mean": "Trung bình số học.",
            "std": "Độ lệch chuẩn mẫu (ddof=1).",
            "min": "Nhỏ nhất.",
            "25%": "Phân vị 25 (Q1).",
            "50%": "Trung vị.",
            "75%": "Phân vị 75 (Q3).",
            "max": "Lớn nhất.",
        }
        alias = {
            "count": ["count", "số lượng"],
            "mean": ["mean", "trung bình"],
            "std": ["std", "độ lệch chuẩn"],
            "min": ["min", "nhỏ nhất"],
            "25%": ["25%", "q1"],
            "50%": ["50%", "median", "trung vị"],
            "75%": ["75%", "q3"],
            "max": ["max", "lớn nhất"],
        }

        def norm(s):
            import re as _re
            return _re.sub(r"\s+", " ", _re.sub(r"<[^>]+>", "", str(s))).strip().lower()

        tips = {}
        for col in ddf.columns:
            c = norm(col)
            for k, names in alias.items():
                if c == k or any(c == norm(n) for n in names):
                    tips[str(col)] = base[k]
                    break
        return tips

    with tab1:
        col1, col2 = st.columns(2)
        if id_col in df.columns:
            col1.metric("Số Tỉnh/Thành", int(df[id_col].nunique()))
        else:
            col1.metric("Số Tỉnh/Thành", len(df))
        crit_cols = [c for c in df.columns if c not in non_crit_cols]
        col2.metric("Số tiêu chí", int(len(crit_cols)))

        st.subheader("Thống kê Mô tả")
        if crit_cols:
            desc = df[crit_cols].describe().T.reset_index().rename(columns={"index": "Tiêu chí"})
            desc["Tiêu chí"] = desc["Tiêu chí"].map(lambda x: name_map.get(x, nice_name(x)))
            _desc_view = apply_display_names(desc)
            display_table(
                _desc_view,
                bold_first_col=True,
                fixed_height=360,
                header_tooltips=_resolve_desc_tooltips(_desc_view),
            )
        else:
            st.info("Không tìm thấy tiêu chí số liệu để thống kê.")

        st.subheader("Bảng Dữ liệu gốc")
        raw = df.copy().rename(columns=name_map)
        if id_col in raw.columns:
            raw[id_col] = raw[id_col].astype(str).str.title()
        display_table(add_index_col(raw, "STT"), bold_first_col=True, fixed_height=420)

    with tab2:
        st.subheader("Chi tiết theo tiêu chí")
        criteria_list = [c for c in df.columns if c not in non_crit_cols]
        if not criteria_list:
            st.info("Không có tiêu chí để phân tích.")
        else:
            cdisp_map = criteria_display_map(criteria_list, metadata)
            options = [cdisp_map[c] for c in criteria_list]
            selected_label = st.selectbox("Chọn tiêu chí:", options, key="data_overview_criterion")
            inv_map = {v: k for k, v in cdisp_map.items()}
            selected_criterion = inv_map[selected_label]

            meta_info = metadata.get(selected_criterion, {})
            full_name = meta_info.get("display_name", nice_name(selected_criterion))
            desc_text = meta_info.get("description", "Không có mô tả.")
            c_type = meta_info.get("type", "N/A")

            st.markdown(f"**{full_name}** · Loại: **{c_type.title()}**")
            st.caption(desc_text)
            st.divider()

            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Top 5 Tỉnh/Thành")
                is_cost = (c_type == "cost")
                sorted_df = df.sort_values(by=selected_criterion, ascending=is_cost).head(5)
                show = sorted_df[[id_col, selected_criterion]].rename(
                    columns={id_col: "Tên Tỉnh/Thành", selected_criterion: full_name}
                )
                display_table(add_index_col(show, "STT"), bold_first_col=True, fixed_height=300)
            with col2:
                st.subheader("Phân phối theo Tỉnh/Thành")
                disp_df = df[[id_col, selected_criterion]].rename(
                    columns={id_col: "Tên Tỉnh/Thành", selected_criterion: full_name}
                )
                chart = alt.Chart(disp_df).mark_bar().encode(
                    x=alt.X("Tên Tỉnh/Thành", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y(full_name),
                    tooltip=["Tên Tỉnh/Thành", full_name],
                ).interactive()
                st.altair_chart(chart, use_container_width=True)


# =========================
# Page: AHP Customize
# =========================
# =========================
# Page: AHP Customize
# =========================
elif page == "Tùy chỉnh Trọng số (AHP)":
    st.header("Trang 3: Tạo và Cập nhật Trọng số Mô hình")

    _SAATY_LABELS = ["1/9","1/8","1/7","1/6","1/5","1/4","1/3","1/2","1","2","3","4","5","6","7","8","9"]
    _SAATY_VALUES = [1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9]
    _VAL_BY_LABEL = {lab: val for lab, val in zip(_SAATY_LABELS, _SAATY_VALUES)}

    def _union_criteria_from_weights(weights_yaml_path: str = F("data/weights.yaml")) -> list[str]:
        opts = set()
        if os.path.exists(weights_yaml_path):
            with open(weights_yaml_path, "r", encoding="utf-8") as f:
                all_w = yaml.safe_load(f) or {}
                if isinstance(all_w, dict):
                    for _, w in all_w.items():
                        if isinstance(w, dict):
                            opts.update(map(str, w.keys()))
        return sorted(opts)

    def _normalize_columns(A):
        A = np.array(A, dtype=float)
        s = A.sum(axis=0); s[s==0] = 1.0
        return A / s

    def _pairwise_matrix_editor(criteria: list[str], session_key: str):
        import pandas as _pd
        n = len(criteria)
        key_df = f"{session_key}_df"
        store_key = f"{session_key}_store"
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
        col_cfg = {nice_name(c): st.column_config.SelectboxColumn(label=nice_name(c), options=_SAATY_LABELS, width="small") for c in criteria}
        st.caption("Nhập trên tam giác trên. Ô '—' = 1. Tam giác dưới tự động nghịch đảo.")
        df_edit = st.data_editor(df, hide_index=False, use_container_width=True, num_rows="fixed", column_config=col_cfg, key=key_df)
        st.session_state[store_key] = df_edit.copy()

        A = np.ones((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j: A[i, j] = 1.0
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
        st.markdown(
            """
<style>
thead tr th div[data-testid="stMarkdownContainer"] p { white-space: nowrap; }
.stDataFrame table { table-layout: fixed; }
.stDataFrame [data-testid="stDataFrameResizable"] { overflow: hidden !important; }
[data-testid="stSegmentedControl"] { max-width: 360px; }
[data-testid="stSegmentedControl"] label { border-radius: 0; min-width: 32px; height: 32px; line-height: 28px; padding: 0; text-align: center; }
.ahp-bar { display: inline-flex; gap: 6px; margin-top: 6px; }
.ahp-bar .sq { width: 14px; height: 14px; border: 1px solid rgba(255,255,255,0.35); }
.ahp-bar .sq.filled { background: rgba(255, 75, 75, 0.55); border-color: rgba(255, 75, 75, 0.75); }
</style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="ahp-bar">{filled}</div>', unsafe_allow_html=True)
        return v

    weights_file = F("data/weights.yaml")
    try:
        with open(weights_file, "r", encoding="utf-8") as f:
            all_weights = yaml.safe_load(f) or {}
    except Exception as e:
        st.error(f"Lỗi đọc weights.yaml: {e}")
        all_weights = {}

    
    st.subheader("1. Lựa chọn Kịch bản (Scenario)")
    model_list = ["Tạo mô hình mới"] + list(all_weights.keys())

    _def_idx = 0
    _after = st.session_state.pop("_after_delete_select", None)
    if _after and _after in model_list:
        _def_idx = model_list.index(_after)

    # Step-2 của wizard tạo mới: ẩn toàn bộ Phần 1
    _wiz = st.session_state.get("ahp_wizard", {})
    _is_new_step2 = (
        st.session_state.get("scenario_selectbox", "Tạo mô hình mới") == "Tạo mô hình mới"
        and isinstance(_wiz, dict) and _wiz.get("step") == 2
    )

    if _is_new_step2:
        selected_scenario = "Tạo mô hình mới"
        st.markdown(
            f"<div class='model-status'>Kịch bản đang tạo: <b style='color:#10b981'>{_wiz.get('name') or 'Custom'}</b></div>",
            unsafe_allow_html=True
        )
    else:
        selected_scenario = st.selectbox(
            "Chọn một kịch bản có sẵn hoặc tạo mới:",
            model_list,
            index=_def_idx,
            key="scenario_selectbox",
        )

        def _protected_scenarios():
            prot = {"Office", "Default", "Baseline"}
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
        with ac1:
            st.button(
                "Tiếp tục tới phân tích (TOPSIS)",
                use_container_width=True,
                disabled=(selected_scenario == "Tạo mô hình mới"),
                key="btn_next_to_topsis",
                on_click=lambda: (
                    st.session_state.update({
                        "selected_model_for_topsis": selected_scenario,
                        "auto_run_topsis": True,
                        "customize_mode": False,
                    }),
                    go("Phân tích Địa điểm (TOPSIS)")
                ),
            )
            if selected_scenario == "Tạo mô hình mới":
                st.caption("Chọn một kịch bản trước khi tiếp tục.")
        with ac2:
            prot = _protected_scenarios()
            can_delete = (selected_scenario not in ("Tạo mô hình mới",)) and (selected_scenario not in prot)
            if st.button("Xóa kịch bản", use_container_width=True, disabled=not can_delete, key="btn_delete_scenario"):
                try:
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
                even = 1.0 / len(new_w)
                new_w = {k: even for k in new_w}
            else:
                new_w = {k: float(v)/s for k, v in new_w.items()}
            try:
                ok, _ = save_weights_to_yaml(new_w, selected_scenario, filename=F("data/weights.yaml"))
            except Exception:
                ok = False
            _notify_saved(ok)
            st.session_state.pop(f"pw_edit_{selected_scenario}", None)
            st.rerun()

    if selected_scenario == "Tạo mô hình mới":
        w = st.session_state.ahp_wizard
        if w["step"] == 1:
            st.subheader("Bước 1: Chọn tiêu chí")
            available = _union_criteria_from_weights()
            ui_cols = st.columns([1,1,4])
            if ui_cols[0].button("Chọn tất cả"):
                for c in available: st.session_state[f"chk_{c}"] = True
            if ui_cols[1].button("Bỏ chọn tất cả"):
                for c in available: st.session_state[f"chk_{c}"] = False

            cols = st.columns(3)
            chosen = set(w.get("selected") or [])
            for idx, c in enumerate(available):
                with cols[idx%3]:
                    chk = st.checkbox(nice_name(c), value=(c in chosen), key=f"chk_{c}")
                    (chosen.add if chk else chosen.discard)(c)

            model_name = st.text_input("Đặt tên mô hình", value=w.get("name") or "Custom")
            st.session_state.ahp_wizard["name"] = model_name
            st.session_state.ahp_wizard["selected"] = sorted(list(chosen))
            st.button("Next", disabled=(len(chosen)<2 or not model_name.strip()), on_click=lambda: (st.session_state.pop("new_model_mode", None), st.session_state.ahp_wizard.update(step=2)))
        else:
            selected = st.session_state.ahp_wizard["selected"]
            model_name = st.session_state.ahp_wizard["name"].strip()
            st.subheader("Bước 2: So sánh/Chấm điểm")
            mode_new = st.radio("Chọn phương thức nhập", ["Pairwise (AHP)", "Direct rating (1–10)"], index=1, horizontal=True, key="new_model_mode")

            if mode_new == "Pairwise (AHP)":
                A_input = _pairwise_matrix_editor(selected, session_key="pw_new_editor")
                M_norm = _normalize_columns(A_input)

                c2 = st.container()
                with c2:
                    if st.button("Lưu + Chạy TOPSIS"):
                        weights_vec, _ = calculate_ahp_weights(M_norm)
                        if weights_vec is not None:
                            s = float(sum(weights_vec)) or 1.0
                            weights_norm = {k: float(v) / s for k, v in zip(selected, weights_vec)}
                            ok, saved_name = save_weights_to_yaml(weights_norm, model_name, filename=F("data/weights.yaml"))
                            _notify_saved(ok)
                            if ok:
                                _run_topsis_from(saved_name or model_name)
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
                if st.button("Lưu + Chạy TOPSIS"):
                    ok, saved_name = save_weights_to_yaml(weights_norm, model_name, filename=F("data/weights.yaml"))
                    _notify_saved(ok)
                    if ok:
                        _run_topsis_from(saved_name or model_name)


            if st.button("Quay lại bước 1"):
                st.session_state.ahp_wizard["step"] = 1; st.rerun()

    else:
        st.subheader(f"Trọng số hiện tại: **{selected_scenario}**")
        current_weights = all_weights.get(selected_scenario, {})
        if current_weights:
            dfw = pd.DataFrame([(nice_name(k), v) for k, v in current_weights.items()],
                               columns=["Tiêu chí","Trọng số"]).sort_values("Trọng số", ascending=False).reset_index(drop=True)
            st.dataframe(dfw, hide_index=True, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Customize")
            _cust_visible_key = f"cust_visible_{selected_scenario}"
            cust_visible = st.toggle("Bật Customize", value=False, key=_cust_visible_key)
            if not cust_visible:
                st.caption("Customize đang tắt.")
                st.stop()

            _ui_delete_features(selected_scenario, current_weights)
            mode = st.radio("Phương thức chỉnh", ["Pairwise (AHP)", "Direct rating (1–10)"], index=1, horizontal=True, key=f"cust_mode_{selected_scenario}")
            features = list(current_weights.keys())

            if mode == "Pairwise (AHP)":
                A_input = _pairwise_matrix_editor(features, session_key=f"pw_edit_{selected_scenario}")
                M_norm = _normalize_columns(A_input)
                weights_vec, _ = calculate_ahp_weights(M_norm)
                if weights_vec is not None:
                    s = float(sum(weights_vec)) or 1.0
                    weights_norm = {k: float(v)/s for k, v in zip(features, weights_vec)}
                    dfp = pd.DataFrame([(nice_name(k), weights_norm[k]) for k in features],
                                       columns=["Tiêu chí","Trọng số"]).sort_values("Trọng số", ascending=False).reset_index(drop=True)
                    st.dataframe(dfp, hide_index=True, use_container_width=True)
                if st.button("Lưu thay đổi", key=f"save_pw_{selected_scenario}"):
                    ok, _ = save_weights_to_yaml(weights_norm, selected_scenario, filename=F("data/weights.yaml"))
                    _notify_saved(ok)

                if st.button("Chạy TOPSIS với mô hình hiện tại", key=f"run_topsis_{selected_scenario}"):
                    _run_topsis_from(selected_scenario)
            else:
                init = {f: max(1, min(10, int(round(float(current_weights.get(f,0))*10)))) for f in features}
                scores = _direct_rating_2col(features, init, f"cust_block_{selected_scenario}")
                s = sum(scores.values()) or 1
                new_w = {f: float(v)/s for f, v in scores.items()}
                dfp = pd.DataFrame([(nice_name(k), scores[k], new_w[k]) for k in features],
                                   columns=["Tiêu chí","Điểm 1–10","Trọng số (chuẩn hoá)"])
                st.dataframe(dfp, hide_index=True, use_container_width=True)
                if st.button("Lưu thay đổi", key=f"save_num_{selected_scenario}"):
                    ok, _ = save_weights_to_yaml(new_w, selected_scenario, filename=F("data/weights.yaml"))
                    _notify_saved(ok)



    _remember_page_state("Tùy chỉnh Trọng số (AHP)", ["ahp_wizard","scenario_selectbox","new_model_mode","chk_","pw_new_editor","pw_edit_","new_block","cust_","del_feats_","save_pw_","save_num_","cust_mode_"])
# =========================
# Page: TOPSIS
# =========================
elif page == "Phân tích Địa điểm (TOPSIS)":
    st.header("Trang 4: Xếp hạng Địa điểm TOPSIS")

    try:
        with open(F("data/weights.yaml"), 'r', encoding='utf-8') as f:
            all_weights = yaml.safe_load(f) or {}
            model_names = list(all_weights.keys())
            if not model_names:
                st.warning("Chưa có mô hình AHP."); st.stop()
    except FileNotFoundError:
        st.error("Thiếu data/weights.yaml."); st.stop()

    # 1) Key selectbox phải có trước khi đọc state
    selectbox_key_topsis = "scenario_selectbox"

    # 2) Handoff và lựa chọn trước đó
    _preselect = st.session_state.pop("model_for_next_page", None)
    _prev_name = st.session_state.get(selectbox_key_topsis)

    if _preselect in model_names:
        idx = model_names.index(_preselect)
        st.success(f"Tự động chọn mô hình '{_preselect}'")
    elif _prev_name in model_names:
        idx = model_names.index(_prev_name)
    else:
        idx = 0

    selected_model = st.selectbox(
        "Chọn mô hình",
        model_names,
        index=idx,
        key=selectbox_key_topsis
    )

    def run_and_display_topsis(model_name: str):
        st.session_state['last_topsis_model'] = model_name

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
                    wdf = wdf.sort_values("Trọng số (%)", ascending=False).reset_index(drop=True)
                    wdf["_label"] = wdf["Trọng số (%)"].round().astype(int).astype(str) + "%"
                    base = alt.Chart(wdf).encode(
                        theta=alt.Theta("Trọng số (%)", stack=True),
                        color=alt.Color("Tiêu chí", sort=wdf["Tiêu chí"].tolist()),
                        order=alt.Order("Trọng số (%)", sort="descending")
                    )
                    pie = base.mark_arc(outerRadius=120, innerRadius=55)
                    text = base.transform_filter(alt.datum["Trọng số (%)"] >= 0).mark_text(radius=148).encode(text=alt.Text("_label:N"))
                    st.altair_chart(pie + text, use_container_width=True)
                except Exception:
                    pass
            st.divider()

        report_df = run_topsis_model(
            data_path=F("AHP_Data_synced_fixed.xlsx"),
            json_path=F("metadata.json"),
            analysis_type=model_name,
            all_criteria_weights=all_weights
        )
        if report_df is not None:
            st.session_state['last_topsis_df'] = report_df.copy()
            st.subheader("Xếp hạng đầy đủ")
            display_table(add_index_col(report_df.copy(), "STT"), bold_first_col=True, fixed_height=380)
            st.divider()
            cols = st.columns(4)
            if cols[0].button("Rerun DSS", use_container_width=True): st.rerun()
            cols[1].button("Map View", on_click=switch_to_map_view, use_container_width=True)
            cols[2].button("Sensitivity", on_click=switch_to_sensitivity, use_container_width=True)
            cols[3].button("Customize AHP", on_click=switch_to_ahp_customize, use_container_width=True)
        else:
            st.error("Lỗi khi phân tích TOPSIS.")

    # 3) Hiển thị ngay nếu đã có kết quả cho mô hình đang chọn
    cached_df = st.session_state.get('last_topsis_df')
    cached_model = st.session_state.get('last_topsis_model')
    if isinstance(cached_df, pd.DataFrame) and not cached_df.empty and cached_model == selected_model:
        run_and_display_topsis(selected_model)  # sẽ chỉ render bảng từ cache và cập nhật nút
    elif st.session_state.pop('topsis_autorun', False):
        run_and_display_topsis(selected_model)
    else:
        if st.button(f"Chạy '{selected_model.upper()}'"):
            run_and_display_topsis(selected_model)

    _remember_page_state("Phân tích Địa điểm (TOPSIS)", ["scenario_selectbox","topsis_model_selector","last_topsis_df","last_topsis_model"])
# =========================
# Page: What-if
# =========================
elif page == "Phân tích Độ nhạy (What-if)":
    st.header("Trang 5: Phân tích Độ nhạy (What-if)")

    # 1) Tải mô hình AHP
    try:
        with open(F("data/weights.yaml"), 'r', encoding='utf-8') as f:
            all_weights = yaml.safe_load(f) or {}
            model_names = list(all_weights.keys())
            if not model_names:
                st.warning("Chưa có mô hình AHP."); st.stop()
    except FileNotFoundError:
        st.error("Thiếu data/weights.yaml."); st.stop()

    # 2) Chọn mô hình: dùng 1 selectbox duy nhất, hỗ trợ handoff
    selectbox_key_whatif = "whatif_model_selector"

    # Handoff ưu tiên: từ TOPSIS → What-if
    _handoff = st.session_state.pop("model_for_whatif", None)
    # Fallback cho code cũ nếu còn dùng khóa này
    if _handoff is None:
        _handoff = st.session_state.pop("model_for_next_page", None)

    _prev = st.session_state.get(selectbox_key_whatif, None)

    if _handoff in model_names:
        idx = model_names.index(_handoff)
        st.success(f"Tự động chọn mô hình '{_handoff}'")
    elif _prev in model_names:
        idx = model_names.index(_prev)
    else:
        idx = 0

    selected_model = st.selectbox(
        "Chọn mô hình gốc:",
        model_names,
        index=idx,
        key=selectbox_key_whatif
    )
    # Hiển thị ngay nếu đã có kết quả What-if cho mô hình này
    _w_cache_model = st.session_state.get("whatif_cached_model")
    _w_orig = st.session_state.get("whatif_original_df")
    _w_new = st.session_state.get("whatif_new_df")
    _w_weights = st.session_state.get("whatif_cached_weights") or {}
    if _w_cache_model == selected_model and isinstance(_w_orig, pd.DataFrame) and isinstance(_w_new, pd.DataFrame) and not _w_orig.empty and not _w_new.empty:
        st.info("Hiển thị kết quả What-if đã lưu. Bấm chạy lại nếu bạn đã thay đổi slider.")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Xếp hạng gốc")
            display_table(add_index_col(_w_orig.copy(), "STT"), True, 420)
        with c2:
            st.subheader("Xếp hạng mới")
            display_table(add_index_col(_w_new.copy(), "STT"), True, 420)
        st.divider()
        st.subheader("So sánh phân bổ trọng số")
        original_weights = all_weights.get(selected_model, {})
        _new_weights_for_chart = _w_weights if _w_weights else original_weights
        st.altair_chart(
            pie_compare_weight(
                original_weights, _new_weights_for_chart,
                title_left="1. Gốc", title_right="2. Mới"
            ),
            use_container_width=True,
        )
        st.subheader("Bảng thay đổi thứ hạng")
        if "last_whatif_rank_changes" in st.session_state and isinstance(st.session_state["last_whatif_rank_changes"], pd.DataFrame):
            display_table(st.session_state["last_whatif_rank_changes"][["Tên phường","Hạng Mới","Hạng Gốc","Thay đổi"]], True, 420)

    # 3) Giao diện điều chỉnh trọng số
    if selected_model:
        original_weights = all_weights.get(selected_model, {})
        st.subheader(f"Điều chỉnh Trọng số — {selected_model.upper()}")

        new_weights = {}
        try:
            df_data = pd.read_excel(F("AHP_Data_synced_fixed.xlsx"))
            full_criteria_list = [c for c in df_data.columns if c not in ["ward", "ward_id", "Tỉnh/Thành phố", "Vùng"]]
        except FileNotFoundError:
            st.error("Thiếu dữ liệu."); st.stop()

        model_criteria = list(original_weights.keys())
        other_criteria = [c for c in full_criteria_list if c not in model_criteria]

        for c in model_criteria:
            new_weights[c] = st.slider(
                nice_name(c), 0.0, 1.0, float(original_weights.get(c, 0.0)), 0.01, key=f"slider_{c}_{selected_model}"
            )

        if other_criteria:
            st.markdown("**Tiêu chí không sử dụng (trọng số = 0)**")
            df_unused = add_index_col(
                pd.DataFrame({"Tiêu chí không sử dụng": [nice_name(c) for c in other_criteria]}),
                "STT"
            )
            display_table(df_unused, bold_first_col=True, fixed_height=240)

        s = float(sum(new_weights.values()))
        normalized_weights = {k: (v / s if s > 0 else 0.0) for k, v in new_weights.items()}
        st.caption(f"Tổng trọng số mới = {s:.2f}. Đã chuẩn hóa." if s > 0 else "Tất cả trọng số đang bằng 0.")

        # 4) Chạy What-if
        if st.button("Chạy What-if"):
            original_df, new_df = run_what_if_analysis(selected_model, normalized_weights, all_weights)
            if original_df is not None and new_df is not None:
                st.session_state["whatif_original_df"] = original_df.copy()
                st.session_state["whatif_new_df"] = new_df.copy()
                st.session_state["whatif_cached_model"] = selected_model
                st.session_state["whatif_cached_weights"] = dict(normalized_weights)
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Xếp hạng gốc")
                    display_table(add_index_col(original_df.copy(), "STT"), True, 420)
                with c2:
                    st.subheader("Xếp hạng mới")
                    display_table(add_index_col(new_df.copy(), "STT"), True, 420)

                st.divider()
                st.subheader("So sánh phân bổ trọng số")
                st.altair_chart(
                    pie_compare_weight(
                        original_weights, normalized_weights,
                        title_left="1. Gốc", title_right="2. Mới"
                    ),
                    use_container_width=True,
                )

                st.subheader("Bảng thay đổi thứ hạng")
                df_orig_simple = original_df[['Tên Tỉnh/Thành', 'Rank']].rename(columns={'Rank': 'Hạng Gốc'})
                df_new_simple = new_df[['Tên Tỉnh/Thành', 'Rank']].rename(columns={'Rank': 'Hạng Mới'})
                df_rank_change = pd.merge(df_orig_simple, df_new_simple, on='Tên Tỉnh/Thành')
                df_rank_change['Thay đổi (số)'] = df_rank_change['Hạng Gốc'] - df_rank_change['Hạng Mới']
                # f-string đúng cho cả số âm
                df_rank_change['Thay đổi'] = df_rank_change['Thay đổi (số)'] \
                    .apply(lambda d: f"▲ +{d}" if d > 0 else (f"▼ {d}" if d < 0 else "—"))
                df_rank_change = df_rank_change.sort_values(by='Hạng Mới')
                st.session_state['last_whatif_rank_changes'] = df_rank_change.copy()
                display_table(df_rank_change[['Tên Tỉnh/Thành', 'Hạng Mới', 'Hạng Gốc', 'Thay đổi']], True, 420)
            else:
                st.error("Lỗi khi chạy What-if.")



    _remember_page_state("Phân tích Độ nhạy (What-if)", ["whatif_","whatif_model_selector","slider_","last_whatif_rank_changes"])
# =========================
# Page: Map View
# =========================
elif page == "Map View":
    st.header("Trang 6: Trực quan bản đồ")

    model_to_map = st.session_state.get('model_for_next_page')
    if not model_to_map:
        st.warning("Cần chạy TOPSIS trước để chọn mô hình."); st.stop()
    st.success(f"Kết quả cho mô hình: **{model_to_map}**")
    st.session_state["last_map_model"] = model_to_map

    # NEW: thêm 'Xanh dương'
    hue_label = st.radio("Màu heatmap", ["Xanh lá", "Đỏ", "Xanh dương"], horizontal=True, key="map_hue")
    hue = {"Xanh lá": "green", "Đỏ": "red", "Xanh dương": "blue"}[hue_label]

    geojson_file = "VietNam_provinces.json"
    ranking_file = f"ranking_result_{model_to_map}.xlsx"

    with open(F(geojson_file), 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    df_ranking = pd.read_excel(F(ranking_file))

    try:
        # Chuẩn hóa tên tỉnh/thành trong file xếp hạng để join với GeoJSON
        ranking_lookup = {
            norm_province_name(r['Tên Tỉnh/Thành']): r.to_dict()
            for _, r in df_ranking.iterrows()
        }
    except KeyError:
        st.error(
            "LỖI: File xếp hạng (Excel) không có cột 'Tên Tỉnh/Thành'. Vui lòng kiểm tra lại file hoặc module TOPSIS.")
        st.stop()

    # --- mở rộng palette rời rạc cho pydeck ---
    def mono_palette(h):
        if h == "red":
            return [[255,235,230,220],[255,210,195,220],[255,184,160,220],[248,150,120,220],[232,112,84,220],[206,72,58,220],[170,40,40,220]]
        if h == "green":
            return [[230,248,235,220],[201,236,214,220],[166,222,189,220],[120,206,160,220],[72,188,128,220],[34,163,98,220],[16,128,74,220]]
        # NEW: blue
        return [[235,244,255,220],[206,229,255,220],[174,214,255,220],[140,197,255,220],[104,178,255,220],[68,153,235,220],[36,120,200,220]]
    PAL = mono_palette(hue); BINS = len(PAL)

    def color_from_score_discrete(s):
        t = 0.0 if s is None else max(0.0, min(1.0, float(s)))
        return PAL[min(BINS - 1, int(round(t * (BINS - 1))))]

    def color_for_rank(rank, score):
        try:
            rnk = int(rank)
        except:
            rnk = 0
        if rnk == 1: return PAL[-1]
        if rnk == 2: return PAL[-2]
        if rnk == 3: return PAL[-3]
        return color_from_score_discrete(score)


    features_found = 0
    # [FIX 2] Thay 'name' thành 'NAME_1' (hoặc key tên tỉnh trong file GeoJSON)
    geojson_key_to_join = 'NAME_1'


    for ftr in geojson_data.get('features', []):
        props = ftr.get('properties', {}) or {}
        name_raw = props.get(geojson_key_to_join)
        if not name_raw:
            continue

        # Chuẩn hóa key join bằng norm_province_name
        key = norm_province_name(name_raw)
        info = ranking_lookup.get(key)

        # Thử lại với VARNAME_1 hoặc name nếu có (một số tỉnh có viết tắt/biệt danh)
        if info is None:
            alt_raw = props.get('VARNAME_1') or props.get('name')
            if alt_raw:
                key_alt = norm_province_name(alt_raw)
                info = ranking_lookup.get(key_alt)

        if info is not None:
            rank = int(info.get('Rank'))
            try:
                score = float(info.get('Điểm TOPSIS (0-1)'))
            except:
                score = 0.0
            ftr['properties'].update(Rank=rank, Score=score, color=color_for_rank(rank, score))
            features_found += 1
        else:
            ftr['properties'].update(Rank="N/A", Score=None, color=PAL[0])  # Tô màu nhạt nhất cho tỉnh không có data

    if features_found == 0:
        st.warning(f"Không thể join bất kỳ tỉnh nào giữa Excel và GeoJSON (dùng key '{geojson_key_to_join}'). "
                   "Hãy kiểm tra sự khác biệt về tên (ví dụ: 'Bà Rịa - Vũng Tàu' vs 'Bà Rịa–Vũng Tàu').")
    else:
        st.info(f"Đã join thành công {features_found} / 63 tỉnh.")

    st.subheader("Bản đồ Xếp hạng TOPSIS")
    view_state = pdk.ViewState(latitude=16.0, longitude=108.0, zoom=5, pitch=0, bearing=0)
    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson_data,
        opacity=0.9,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="properties.color",
        get_line_color=[25, 25, 25, 160],
        get_line_width=60,
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        "html": f"<b>Tỉnh:</b> {{{geojson_key_to_join}}}<br/><b>Hạng:</b> {{Rank}}<br/><b>Điểm TOPSIS:</b> {{Score}}",
        "style": {"backgroundColor": "steelblue", "color": "white"}}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style=pdk.map_styles.LIGHT, tooltip=tooltip), use_container_width=True)

    
    
    
    # NEW: nút xuất PDF đồng bộ màu Map View
    c1, c2 = st.columns([1,3])
    with c1:
        ph = st.empty()
        with ph.container():
            if 'create_full_report' in globals():
                # Trạng thái xuất để tránh trắng màn hình
                exporting_key = "map_exporting"
                if st.session_state.get(exporting_key, False):
                    try:
                        with st.status("Đang xuất PDF...", expanded=True) as status:
                            with open(F("data/weights.yaml"), "r", encoding="utf-8") as f:
                                all_weights = yaml.safe_load(f) or {}
                            pdf_path = create_full_report(model_to_map, all_weights, hue=hue)
                            if pdf_path and os.path.exists(pdf_path):
                                status.update(label="Xuất xong", state="complete")
                                st.success("Đã xuất PDF.")
                                st.caption(pdf_path)
                            else:
                                status.update(label="Xuất thất bại", state="error")
                                st.error("Xuất PDF thất bại.")
                    except Exception as e:
                        st.error(f"Lỗi tạo PDF: {e}")
                    finally:
                        st.session_state[exporting_key] = False
                else:
                    if st.button("Xuất PDF", use_container_width=True, key="export_map_pdf"):
                        st.session_state[exporting_key] = True
                        st.rerun()
            else:
                st.caption("Thiếu report_module, không thể xuất PDF.")




    _remember_page_state("Map View", ["map_","map_hue", "model_for_next_page"])
