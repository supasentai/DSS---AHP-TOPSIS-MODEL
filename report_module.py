# report_module.py — cleaned
import os
import re
import json
import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas
from fpdf import FPDF

# =========================
# Paths
# =========================
WEIGHTS_PATH = "data/weights.yaml"
METADATA_PATH = "data/metadata.json"
DATA_PATH = "data/AHP_Data_synced_fixed.xlsx"
GEOJSON_PATH = "data/quan7_geojson.json"
RANKING_DIR = "data"
REPORT_OUTPUT_DIR = "reports"
TEMP_ASSET_DIR = os.path.join(REPORT_OUTPUT_DIR, "temp_assets")

# =========================
# Palette
# =========================
PALETTE = {
    "red": "#F94144",
    "orange": "#F3722C",
    "amber": "#F8961E",
    "yellow": "#F9C74F",
    "green": "#90BE6D",
    "teal": "#43AA8B",
    "blue": "#577590",
}

def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

# =========================
# Matplotlib font
# =========================
try:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    FONT_NAME = "DejaVu"
    FONT_PATH = "Font"  # chứa DejaVuSans*.ttf
except Exception:
    plt.rcParams["font.family"] = "Arial"
    FONT_NAME = "Arial"
    FONT_PATH = ""

# =========================
# PDF helper
# =========================
class PDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "N/A"
        self.active_font = FONT_NAME
        if FONT_NAME == "DejaVu":
            try:
                self.add_font(self.active_font, "", os.path.join(FONT_PATH, "DejaVuSans.ttf"), uni=True)
                self.add_font(self.active_font, "B", os.path.join(FONT_PATH, "DejaVuSans-Bold.ttf"), uni=True)
                self.add_font(self.active_font, "I", os.path.join(FONT_PATH, "DejaVuSans-Oblique.ttf"), uni=True)
            except RuntimeError:
                self.active_font = "Arial"

    def set_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    def header(self):
        self.set_font(self.active_font, "B", 15)
        r, g, b = _hex_to_rgb(PALETTE["blue"])
        self.set_text_color(r, g, b)
        self.cell(0, 10, "Báo cáo Hệ thống Hỗ trợ Quyết định (DSS)", 0, 1, "C")
        self.set_font(self.active_font, "", 12)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, f"Mô hình Phân tích: {self.model_name}", 0, 1, "C")
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.active_font, "I", 8)
        self.cell(0, 10, f"Trang {self.page_no()}/{{nb}}", 0, 0, "C")
        self.set_x(-60)
        self.cell(0, 10, f"Ngày: {datetime.date.today().isoformat()}", 0, 0, "R")

    def section_title(self, title: str) -> None:
        self.set_font(self.active_font, "B", 14)
        r, g, b = _hex_to_rgb(PALETTE["orange"])
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, f" {title}", 0, 1, "L", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def add_df_to_table(self, df: pd.DataFrame, col_widths=None, highlight_top: bool = False) -> None:
        self.set_font(self.active_font, "B", 10)
        if col_widths is None:
            w = (self.w - 2 * self.l_margin) / max(1, len(df.columns))
            col_widths = [w] * len(df.columns)

        # header
        self.set_fill_color(240, 240, 240)
        self.set_text_color(0, 0, 0)
        self.set_draw_color(220, 220, 220)
        for i, col_name in enumerate(df.columns):
            self.cell(col_widths[i], 8, str(col_name), border=1, fill=True, align="C")
        self.ln()

        # body
        self.set_font(self.active_font, "", 9)
        c_rank1 = _hex_to_rgb(PALETTE["red"])
        c_rank2 = _hex_to_rgb(PALETTE["orange"])
        c_rank3 = _hex_to_rgb(PALETTE["amber"])

        even = False
        for _, row in df.iterrows():
            even = not even
            self.set_fill_color(248, 248, 248) if even else self.set_fill_color(255, 255, 255)

            if highlight_top and "Rank" in row.index:
                r = row["Rank"]
                if r == 1:
                    self.set_text_color(*c_rank1)
                    self.set_font(self.active_font, "B", 9)
                elif r == 2:
                    self.set_text_color(*c_rank2)
                    self.set_font(self.active_font, "B", 9)
                elif r == 3:
                    self.set_text_color(*c_rank3)
                    self.set_font(self.active_font, "B", 9)
                else:
                    self.set_text_color(0, 0, 0)
                    self.set_font(self.active_font, "", 9)

            for i, item in enumerate(row):
                s = f"{item:.4f}" if isinstance(item, float) else str(item)
                self.cell(col_widths[i], 7, s, border=1, fill=True, align="L")

            self.set_text_color(0, 0, 0)
            self.set_font(self.active_font, "", 9)
            self.ln()
        self.ln(4)

# =========================
# Data helpers
# =========================
def _nice_name(col: str) -> str:
    return str(col).replace("_", " ").strip().title()

def _load_all_data(model_name: str, all_weights: Dict) -> Optional[Tuple[Dict, pd.DataFrame, pd.DataFrame, Dict]]:
    try:
        weights_dict = all_weights.get(model_name)
        if not weights_dict:
            print(f"Không tìm thấy mô hình '{model_name}'")
            return None

        ranking_file = os.path.join(RANKING_DIR, f"ranking_result_{model_name}.xlsx")
        ranking_df = pd.read_excel(ranking_file)

        raw_data_df = pd.read_excel(DATA_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8-sig") as f:
            metadata = json.load(f)

        return weights_dict, ranking_df, raw_data_df, metadata
    except Exception as e:
        print(f"Lỗi tải dữ liệu: {e}")
        return None

# =========================
# Charts
# =========================
def _generate_weights_pie_chart(weights_dict: Dict, output_path: str) -> Optional[str]:
    try:
        sorted_items = sorted(weights_dict.items(), key=lambda kv: kv[1], reverse=True)
        labels = [_nice_name(k) for k, _ in sorted_items]
        sizes = [float(v) for _, v in sorted_items]
        sizes = [s for s in sizes if s > 0]
        labels = [l for l, s in zip(labels, [float(v) for _, v in sorted_items]) if s > 0]
        if not sizes:
            return None

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = [PALETTE[c] for c in ["red", "orange", "amber", "yellow", "green", "teal", "blue"]][: len(sizes)]
        wedges, _, autotexts = ax.pie(
            sizes,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.6,
            colors=colors,
        )
        ax.axis("equal")
        ax.legend(wedges, labels, title="Tiêu chí", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=10, weight="bold", color="black")
        ax.set_title("Phân bổ Trọng số Tiêu chí (AHP)", size=16, weight="bold")

        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return output_path
    except Exception as e:
        print(f"Lỗi vẽ pie: {e}")
        return None

def _generate_radar_chart(ranking_df: pd.DataFrame,
                          raw_data_df: pd.DataFrame,
                          metadata: Dict,
                          output_path: str) -> Optional[str]:
    try:
        keys = ["population_density", "rental_cost", "competition_pressure", "amenity_score"]
        types = {k: v.get("type") for k, v in metadata.items()}

        # labels
        radar_labels = []
        for k in keys:
            if k in raw_data_df.columns:
                tag = "(Lợi ích)" if types.get(k) == "benefit" else "(Chi phí)"
                radar_labels.append(f"{_nice_name(k)}\n{tag}")

        # normalize 0..1; cost → invert
        norm_df = raw_data_df.copy()
        for c in keys:
            if c not in norm_df.columns:
                continue
            mn, mx = norm_df[c].min(), norm_df[c].max()
            rng = (mx - mn) or 1.0
            norm_df[c] = (norm_df[c] - mn) / rng
            if types.get(c) == "cost":
                norm_df[c] = 1.0 - norm_df[c]

        top_names = ranking_df.sort_values("Rank")["Tên phường"].tolist()[:3]
        colors = [PALETTE["orange"], PALETTE["green"], PALETTE["teal"]]

        angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for i, name in enumerate(top_names):
            row = norm_df[norm_df["ward"] == name]
            if row.empty:
                continue
            vals = [row.iloc[0].get(k, 0.0) for k in keys if k in norm_df.columns]
            if not vals:
                continue
            vals += vals[:1]
            ax.plot(angles, vals, color=colors[i], linewidth=2, label=name)
            ax.fill(angles, vals, color=colors[i], alpha=0.2)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        ax.set_title("Hồ sơ Top 3 Phường", size=16, weight="bold", y=1.1)

        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return output_path
    except Exception as e:
        print(f"Lỗi vẽ radar: {e}")
        plt.close("all")
        return None

def _generate_bar_charts(raw_data_df: pd.DataFrame, output_prefix: str) -> Dict[str, str]:
    try:
        crits = {
            "population_density": PALETTE["teal"],
            "rental_cost": PALETTE["orange"],
            "competition_pressure": PALETTE["red"],
        }
        wards = raw_data_df["ward"].astype(str).tolist()
        paths: Dict[str, str] = {}

        for key, color in crits.items():
            if key not in raw_data_df.columns:
                continue
            vals = raw_data_df[key].tolist()
            out = f"{output_prefix}_{key}.png"

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(wards, vals, color=color)
            ax.set_title(f"Phân tích Tiêu chí: {_nice_name(key)}", size=16, weight="bold")
            ax.set_ylabel("Giá trị gốc")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(out, bbox_inches="tight", dpi=150)
            plt.close(fig)
            paths[key] = out

        return paths
    except Exception as e:
        print(f"Lỗi vẽ bar: {e}")
        return {}

def _generate_ranking_map(ranking_df: pd.DataFrame, geojson_path: str, output_path: str, hue: str = "blue") -> Optional[str]:
    try:
        gdf = geopandas.read_file(geojson_path)
        gdf["join_key"] = gdf["name"].astype(str).str.replace(" ", "")
        tmp = ranking_df.copy()
        tmp["join_key"] = tmp["Tên phường"].astype(str).str.replace(" ", "")
        merged = gdf.merge(tmp, on="join_key", how="left")
        merged["Điểm TOPSIS (0-1)"] = merged["Điểm TOPSIS (0-1)"].fillna(0.0)

        # NEW: đồng bộ colormap theo hue
        cmap = {"red": "Reds", "green": "Greens", "blue": "Blues"}.get(str(hue).lower(), "Blues")

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        merged.plot(
            column="Điểm TOPSIS (0-1)",
            ax=ax,
            legend=True,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            missing_kwds={"color": "lightgrey", "label": "Không có dữ liệu"},
            legend_kwds={"label": "Điểm TOPSIS (0–1)", "orientation": "horizontal"},
        )
        for x, y, label in zip(merged.geometry.centroid.x, merged.geometry.centroid.y, merged["name"]):
            ax.text(x, y, label, fontsize=8, ha="center", color="black")

        ax.set_title("Bản đồ Xếp hạng TOPSIS Quận 7", size=18, weight="bold")
        ax.axis("off")
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return output_path
    except Exception as e:
        print(f"Lỗi vẽ bản đồ: {e}")
        return None

# =========================
# Main API
# =========================
def create_full_report(model_name: str, all_weights: Dict, hue: str = "blue") -> Optional[str]:
    """
    Tạo báo cáo PDF gồm:
      - Pie trọng số AHP
      - Bảng xếp hạng TOPSIS
      - Radar Top 3
      - Bar theo tiêu chí
      - Bản đồ tĩnh
    """
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_ASSET_DIR, exist_ok=True)

    data = _load_all_data(model_name, all_weights)
    if data is None:
        return None
    weights_dict, ranking_df, raw_data_df, metadata = data

    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
    def _tmp(name: str) -> str:
        return os.path.join(TEMP_ASSET_DIR, f"{safe}_{name}.png")

    pie_path = _generate_weights_pie_chart(weights_dict, _tmp("weights_pie"))
    radar_path = _generate_radar_chart(ranking_df, raw_data_df, metadata, _tmp("radar_top3"))
    map_path   = _generate_ranking_map(ranking_df, GEOJSON_PATH, _tmp("ranking_map"), hue=hue)
    bar_paths  = _generate_bar_charts(raw_data_df, os.path.join(TEMP_ASSET_DIR, f"{safe}_bar"))

    pdf = PDF("P", "mm", "A4")
    pdf.set_model_name(model_name)
    pdf.alias_nb_pages()

    # Cover
    pdf.add_page()
    pdf.set_font(pdf.active_font, "B", 24)
    pdf.cell(0, 45, "", 0, 1)
    pdf.cell(0, 14, "BÁO CÁO PHÂN TÍCH HỖ TRỢ QUYẾT ĐỊNH", 0, 1, "C")
    pdf.set_font(pdf.active_font, "B", 20)
    pdf.cell(0, 14, f"Mô hình: {model_name}", 0, 1, "C")
    pdf.set_font(pdf.active_font, "", 14)
    pdf.cell(0, 10, f"Ngày tạo: {datetime.date.today().isoformat()}", 0, 1, "C")

    # Page 2: weights
    pdf.add_page()
    pdf.section_title("1. Phân bổ Trọng số (AHP)")
    page_w = pdf.w - 2 * pdf.l_margin
    if pie_path:
        pdf.image(pie_path, x=pdf.l_margin, y=None, w=page_w * 0.9)
        pdf.ln(4)

    pdf.set_font(pdf.active_font, "B", 12)
    pdf.cell(0, 10, "Bảng Trọng số chi tiết:", 0, 1, "L")
    wdf = pd.DataFrame(weights_dict.items(), columns=["Tiêu chí (gốc)", "Trọng số"])
    wdf["Tiêu chí"] = wdf["Tiêu chí (gốc)"].apply(_nice_name)
    wdf = wdf[["Tiêu chí", "Trọng số"]].sort_values("Trọng số", ascending=False)
    pdf.add_df_to_table(wdf, col_widths=[page_w * 0.7, page_w * 0.3])

    # Page 3: ranking
    pdf.add_page()
    pdf.section_title("2. Kết quả Xếp hạng (TOPSIS)")
    keep = [c for c in ["Rank", "Tên phường", "Điểm TOPSIS (0-1)"] if c in ranking_df.columns]
    tbl = ranking_df.sort_values("Rank")[keep]
    pdf.add_df_to_table(tbl, col_widths=[page_w * 0.15, page_w * 0.55, page_w * 0.3], highlight_top=True)

    # Page 4: radar
    pdf.add_page()
    pdf.section_title("3. Hồ sơ Top 3 Phường (Biểu đồ Radar)")
    if radar_path:
        pdf.image(radar_path, x=pdf.l_margin, y=None, w=page_w)
    else:
        pdf.cell(0, 10, "(Không thể tạo biểu đồ radar)", 0, 1, "L")

    # Page 5–6: bars
    pdf.add_page()
    pdf.section_title("4. Phân tích Tiêu chí Chi tiết (Giá trị gốc)")
    for key in ["population_density", "rental_cost", "competition_pressure"]:
        path = bar_paths.get(key)
        if path:
            pdf.image(path, x=pdf.l_margin, y=None, w=page_w)
            pdf.ln(4)
            # tách trang cho đồ thị tiếp theo nếu còn
            if key != "competition_pressure":
                pdf.add_page()

    # Page 7: map
    pdf.add_page()
    pdf.section_title("5. Trực quan Bản đồ Xếp hạng")
    if map_path:
        pdf.image(map_path, x=pdf.l_margin, y=None, w=page_w)
    else:
        pdf.cell(0, 10, "(Không thể tạo bản đồ)", 0, 1, "L")

    # Save + cleanup
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    out_name = f"Bao_cao_PDF_{safe}_{datetime.date.today().isoformat()}.pdf"
    out_path = os.path.join(REPORT_OUTPUT_DIR, out_name)
    try:
        pdf.output(out_path)
    except Exception as e:
        print(f"Lỗi lưu PDF: {e}")
        return None

    for f in [pie_path, radar_path, map_path, *bar_paths.values()]:
        try:
            if f and os.path.exists(f):
                os.remove(f)
        except Exception:
            pass

    return out_path
