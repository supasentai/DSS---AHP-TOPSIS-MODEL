# report_module.py (Phiên bản PDF Nâng cấp)
"""
Module để tạo báo cáo PDF tĩnh, được thiết kế lại
để trông chuyên nghiệp hơn, lấy cảm hứng từ mẫu HTML.

Sử dụng fpdf2 và matplotlib.
"""

import os
import datetime
import pandas as pd
import yaml
import json
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import re
from fpdf import FPDF

# --- CẤU HÌNH ---
# Thêm bảng màu từ mẫu HTML
PALETTE = {
    'red': '#F94144',
    'orange': '#F3722C',
    'amber': '#F8961E',
    'yellow': '#F9C74F',
    'green': '#90BE6D',
    'teal': '#43AA8B',
    'blue': '#577590'
}

# Cấu hình font
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    FONT_NAME = 'DejaVu'
    FONT_PATH = "Font/"  # Thư mục chứa font .ttf
except Exception:
    print("Cảnh báo: Không tìm thấy font 'DejaVu Sans'. Dùng 'Arial'.")
    plt.rcParams['font.family'] = 'Arial'
    FONT_NAME = 'Arial'
    FONT_PATH = ""  # Không cần đường dẫn nếu dùng font cơ bản

# Đường dẫn
WEIGHTS_PATH = "data/weights.yaml"
METADATA_PATH = "data/metadata.json"
DATA_PATH = "data/AHP_Data_synced_fixed.xlsx"
GEOJSON_PATH = "data/quan7_geojson.json"
RANKING_DIR = "data"
REPORT_OUTPUT_DIR = "reports"
TEMP_ASSET_DIR = os.path.join(REPORT_OUTPUT_DIR, "temp_assets")


class PDF(FPDF):
    """
    Lớp PDF tùy chỉnh với Header, Footer và Bảng biểu được làm đẹp
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "N/A"
        self.active_font = FONT_NAME
        try:
            # Đăng ký font hỗ trợ Tiếng Việt
            self.add_font(self.active_font, '', os.path.join(FONT_PATH, 'DejaVuSans.ttf'), uni=True)
            self.add_font(self.active_font, 'B', os.path.join(FONT_PATH, 'DejaVuSans-Bold.ttf'), uni=True)
            self.add_font(self.active_font, 'I', os.path.join(FONT_PATH, 'DejaVuSans-Oblique.ttf'), uni=True)
        except RuntimeError:
            print(f"LỖI FPDF: Không thể tải font 'DejaVuSans' từ '{FONT_PATH}'.")
            self.active_font = 'Arial'  # Dùng font dự phòng

    def set_model_name(self, model_name):
        self.model_name = model_name

    def header(self):
        self.set_font(self.active_font, 'B', 15)
        self.set_text_color(int(PALETTE['blue'][1:3], 16), int(PALETTE['blue'][3:5], 16), int(PALETTE['blue'][5:7], 16))
        self.cell(0, 10, 'Báo cáo Hệ thống Hỗ trợ Quyết định (DSS)', 0, 1, 'C')
        self.set_font(self.active_font, '', 12)
        self.cell(0, 8, f'Mô hình Phân tích: {self.model_name}', 0, 1, 'C')
        self.set_text_color(0, 0, 0)  # Reset màu
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.active_font, 'I', 8)
        self.cell(0, 10, f'Trang {self.page_no()}/{{nb}}', 0, 0, 'C')
        self.set_x(-60)
        self.cell(0, 10, f'Ngày: {datetime.date.today().isoformat()}', 0, 0, 'R')

    def section_title(self, title):
        self.set_font(self.active_font, 'B', 14)
        # Dùng màu cam cho tiêu đề
        self.set_fill_color(int(PALETTE['orange'][1:3], 16), int(PALETTE['orange'][3:5], 16),
                            int(PALETTE['orange'][5:7], 16))
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, f' {title}', 0, 1, 'L', fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(5)

    def add_df_to_table(self, df, col_widths=None, highlight_top=False):
        self.set_font(self.active_font, 'B', 10)

        if col_widths is None:
            num_cols = len(df.columns)
            equal_width = (self.w - 2 * self.l_margin) / num_cols
            col_widths = [equal_width] * num_cols

        # --- Header (Làm đẹp) ---
        self.set_fill_color(240, 240, 240)
        self.set_text_color(0, 0, 0)
        self.set_draw_color(220, 220, 220)
        for i, col_name in enumerate(df.columns):
            self.cell(col_widths[i], 8, str(col_name), border=1, fill=True, align='C')
        self.ln()

        # --- Body ---
        self.set_font(self.active_font, '', 9)
        self.set_fill_color(255, 255, 255)
        is_even_row = False

        # Màu highlight
        c_rank1 = (int(PALETTE['red'][1:3], 16), int(PALETTE['red'][3:5], 16), int(PALETTE['red'][5:7], 16))
        c_rank2 = (int(PALETTE['orange'][1:3], 16), int(PALETTE['orange'][3:5], 16), int(PALETTE['orange'][5:7], 16))
        c_rank3 = (int(PALETTE['amber'][1:3], 16), int(PALETTE['amber'][3:5], 16), int(PALETTE['amber'][5:7], 16))

        for _, row in df.iterrows():
            is_even_row = not is_even_row
            fill_row = is_even_row

            # Logic highlight
            rank = None
            if highlight_top and 'Rank' in row:
                rank = row['Rank']
                if rank == 1:
                    self.set_text_color(*c_rank1)
                    self.set_font(self.active_font, 'B', 9)
                elif rank == 2:
                    self.set_text_color(*c_rank2)
                    self.set_font(self.active_font, 'B', 9)
                elif rank == 3:
                    self.set_text_color(*c_rank3)
                    self.set_font(self.active_font, 'B', 9)
                else:
                    self.set_text_color(0, 0, 0)
                    self.set_font(self.active_font, '', 9)

            self.set_fill_color(248, 248, 248) if fill_row else self.set_fill_color(255, 255, 255)

            for i, item in enumerate(row):
                if isinstance(item, float):
                    item_str = f"{item:.4f}"
                else:
                    item_str = str(item)
                self.cell(col_widths[i], 7, item_str, border=1, fill=True, align='L')

            # Reset font/color
            self.set_text_color(0, 0, 0)
            self.set_font(self.active_font, '', 9)
            self.ln()

        self.ln(5)


# --- CÁC HÀM XỬ LÝ DỮ LIỆU ---

def _nice_name(col: str) -> str:
    return str(col).replace("_", " ").strip().title()


def _load_all_data(model_name, all_weights):
    """Tải tất cả các nguồn dữ liệu cần thiết."""
    try:
        weights_dict = all_weights.get(model_name)
        if not weights_dict:
            print(f"LỖI: Không tìm thấy mô hình '{model_name}'")
            return None

        ranking_file = os.path.join(RANKING_DIR, f"ranking_result_{model_name}.xlsx")
        try:
            ranking_df = pd.read_excel(ranking_file)
        except FileNotFoundError:
            print(f"LỖI: Thiếu file xếp hạng {ranking_file}. Hãy chạy TOPSIS trước.")
            return None

        raw_data_df = pd.read_excel(DATA_PATH)

        with open(METADATA_PATH, 'r', encoding='utf-8-sig') as f:
            metadata = json.load(f)

        return weights_dict, ranking_df, raw_data_df, metadata

    except Exception as e:
        print(f"LỖI khi tải dữ liệu: {e}")
        return None


# --- CÁC HÀM TẠO BIỂU ĐỒ (MATPLOTLIB) ---

def _generate_weights_pie_chart(weights_dict, output_path):
    """Tạo biểu đồ tròn từ dict trọng số (ĐÃ LÀM ĐẸP)"""
    try:
        sorted_weights = sorted(weights_dict.items(), key=lambda item: item[1], reverse=True)
        labels = [_nice_name(k) for k, v in sorted_weights]
        sizes = [v for k, v in sorted_weights]

        # Chỉ vẽ các mục có trọng số > 0
        plot_labels = [l for l, s in zip(labels, sizes) if s > 0]
        plot_sizes = [s for s in sizes if s > 0]

        if not plot_sizes:
            print("Không có trọng số > 0 để vẽ biểu đồ.")
            return None

        fig, ax = plt.subplots(figsize=(10, 7))
        # Lấy màu từ palette
        colors = [PALETTE[c] for c in ['red', 'orange', 'amber', 'yellow', 'green', 'teal', 'blue']]

        wedges, texts, autotexts = ax.pie(
            plot_sizes,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.6,  # Di chuyển % vào trong
            colors=colors[:len(plot_sizes)]  # Dùng màu palette
        )
        ax.axis('equal')

        ax.legend(wedges, plot_labels, title="Tiêu chí", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=10, weight="bold", color="black")
        ax.set_title("Phân bổ Trọng số Tiêu chí (AHP)", size=16, weight="bold")

        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Đã lưu biểu đồ trọng số tại: {output_path}")
        return output_path
    except Exception as e:
        print(f"LỖI khi tạo biểu đồ trọng số: {e}")
        return None


def _generate_radar_chart(ranking_df, raw_data_df, metadata, output_path):
    """Tạo biểu đồ radar cho Top 3 (MỚI)"""
    try:
        # Lấy 4 tiêu chí cốt lõi (giống HTML)
        radar_criteria_keys = ['population_density', 'rental_cost', 'competition_pressure', 'amenity_score']

        # Lấy nhãn đẹp
        radar_labels = []
        types = {k: v.get('type') for k, v in metadata.items()}
        for k in radar_criteria_keys:
            if k not in raw_data_df.columns:
                print(f"Cảnh báo: Thiếu cột '{k}' cho biểu đồ radar.")
                continue
            label = _nice_name(k)
            type_label = "(Lợi ích)" if types.get(k) == 'benefit' else "(Chi phí)"
            radar_labels.append(f"{label}\n{type_label}")

        # Chuẩn hóa dữ liệu cho Radar (1.0 luôn là tốt nhất)
        norm_df = raw_data_df.copy()
        for c in radar_criteria_keys:
            if c in norm_df:
                min_val, max_val = norm_df[c].min(), norm_df[c].max()
                range_val = max_val - min_val if max_val - min_val != 0 else 1
                norm_df[c] = (norm_df[c] - min_val) / range_val
                if types.get(c) == 'cost':
                    norm_df[c] = 1.0 - norm_df[c]

        # Lấy dữ liệu Top 3
        top_names = ranking_df.sort_values('Rank')['Tên phường'].tolist()
        top_3_names = top_names[:3]

        palette_colors = [PALETTE['orange'], PALETTE['green'], PALETTE['teal']]

        # --- Vẽ bằng Matplotlib ---
        angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
        angles += angles[:1]  # Đóng vòng

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for i, name in enumerate(top_3_names):
            ward_data = norm_df[norm_df['ward'] == name]
            if ward_data.empty: continue

            scores = [ward_data.iloc[0].get(k, 0.0) for k in radar_criteria_keys]
            scores += scores[:1]  # Đóng vòng

            ax.plot(angles, scores, color=palette_colors[i], linewidth=2, label=name)
            ax.fill(angles, scores, color=palette_colors[i], alpha=0.2)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title("Hồ sơ Top 3 Phường", size=16, weight="bold", y=1.1)

        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Đã lưu biểu đồ radar tại: {output_path}")
        return output_path

    except Exception as e:
        print(f"LỖI khi tạo biểu đồ radar: {e}")
        import traceback
        traceback.print_exc()
        return None


def _generate_bar_charts(raw_data_df, output_path_prefix):
    """Tạo các biểu đồ cột chi tiết (MỚI)"""
    try:
        criteria_to_plot = {
            'population_density': PALETTE['teal'],
            'rental_cost': PALETTE['orange'],
            'competition_pressure': PALETTE['red']
        }

        wards = raw_data_df['ward'].tolist()
        paths = {}

        for key, color in criteria_to_plot.items():
            if key not in raw_data_df.columns:
                print(f"Cảnh báo: Thiếu cột '{key}' cho biểu đồ cột.")
                continue

            values = raw_data_df[key].tolist()
            output_path = f"{output_path_prefix}_{key}.png"

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(wards, values, color=color)

            ax.set_title(f"Phân tích Tiêu chí: {_nice_name(key)}", size=16, weight="bold")
            ax.set_ylabel("Giá trị gốc")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"Đã lưu biểu đồ cột '{key}' tại: {output_path}")
            paths[key] = output_path

        return paths

    except Exception as e:
        print(f"LỖI khi tạo biểu đồ cột: {e}")
        return {}


def _generate_ranking_map(ranking_df, geojson_path, output_path):
    """
    Tạo bảng đồ tĩnh GeoJSON
    *** CẬP NHẬT: Tô màu theo Điểm TOPSIS (0-1) và dùng gam màu Xanh Dương (Blues) ***
    """
    try:
        gdf = geopandas.read_file(geojson_path)
        gdf['join_key'] = gdf['name'].astype(str).str.replace(" ", "")
        ranking_df_copy = ranking_df.copy()  # Tránh lỗi
        ranking_df_copy['join_key'] = ranking_df_copy['Tên phường'].astype(str).str.replace(" ", "")

        merged_gdf = gdf.merge(ranking_df_copy, on='join_key', how='left')

        # Xử lý các phường không có dữ liệu (nếu có)
        merged_gdf['Điểm TOPSIS (0-1)'] = merged_gdf['Điểm TOPSIS (0-1)'].fillna(0)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # --- THAY ĐỔI LOGIC VẼ ---
        merged_gdf.plot(
            column='Điểm TOPSIS (0-1)',  # Cột để tô màu
            ax=ax,
            legend=True,
            cmap='Blues',  # Gam màu xanh dương (càng xanh đậm càng tốt)
            vmin=0.0,  # Cố định thang màu từ 0
            vmax=1.0,  # Cố định thang màu đến 1
            missing_kwds={
                "color": "lightgrey",
                "label": "Không có dữ liệu",
            },
            legend_kwds={'label': "Điểm TOPSIS (Giá trị 0-1)",  # Cập nhật nhãn
                         'orientation': "horizontal"}
        )
        # --- KẾT THÚC THAY ĐỔI ---

        # Thêm nhãn
        # Cảnh báo UserWarning về centroid là bình thường, có thể bỏ qua
        for x, y, label in zip(merged_gdf.geometry.centroid.x, merged_gdf.geometry.centroid.y, merged_gdf["name"]):
            ax.text(x, y, label, fontsize=8, ha='center', color='black')

        ax.set_title("Bảng đồ Xếp hạng TOPSIS Quận 7", size=18, weight="bold")
        ax.axis('off')

        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Đã lưu bản đồ tại: {output_path}")
        return output_path
    except Exception as e:
        print(f"LỖI khi tạo bản đồ: {e}")
        return None


# --- HÀM CHÍNH TẠO BÁO CÁO ---

def create_full_report(model_name: str, all_weights: dict) -> str | None:
    """
    Hàm chính để tạo báo cáo PDF tổng hợp (ĐÃ NÂNG CẤP).
    """
    print(f"--- Bắt đầu tạo báo cáo PDF Nâng cấp cho: {model_name} ---")

    # --- 0. Chuẩn bị thư mục ---
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_ASSET_DIR, exist_ok=True)

    # --- 1. Tải dữ liệu ---
    data_load = _load_all_data(model_name, all_weights)
    if data_load is None:
        return None
    weights_dict, ranking_df, raw_data_df, metadata = data_load

    # --- 2. Tạo các tài sản (ảnh) ---
    print("Đang tạo tài sản trực quan (biểu đồ, bản đồ)...")

    # Tạo tên file tạm
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)

    def temp_path(suffix):
        return os.path.join(TEMP_ASSET_DIR, f"{safe_name}_{suffix}.png")

    pie_chart_path = _generate_weights_pie_chart(weights_dict, temp_path("weights_pie"))
    radar_chart_path = _generate_radar_chart(ranking_df, raw_data_df, metadata, temp_path("radar_top3"))
    map_image_path = _generate_ranking_map(ranking_df, GEOJSON_PATH, temp_path("ranking_map"))
    bar_chart_paths = _generate_bar_charts(raw_data_df, os.path.join(TEMP_ASSET_DIR, f"{safe_name}_bar"))

    # --- 3. Tạo PDF ---
    print("Đang tổng hợp file PDF...")
    pdf = PDF('P', 'mm', 'A4')
    pdf.set_model_name(model_name)
    pdf.alias_nb_pages()

    # Trang bìa
    pdf.add_page()
    pdf.set_font(pdf.active_font, 'B', 24)
    pdf.cell(0, 50, '', 0, 1)
    pdf.cell(0, 15, 'BÁO CÁO PHÂN TÍCH HỖ TRỢ QUYẾT ĐỊNH', 0, 1, 'C')
    pdf.set_font(pdf.active_font, 'B', 20)
    pdf.cell(0, 15, f'Mô hình: {model_name}', 0, 1, 'C')
    pdf.set_font(pdf.active_font, '', 14)
    pdf.cell(0, 10, f'Ngày tạo: {datetime.date.today().isoformat()}', 0, 1, 'C')

    # Trang 2: Trọng số AHP
    pdf.add_page()
    pdf.section_title('1. Phân bổ Trọng số (AHP)')

    page_width = pdf.w - 2 * pdf.l_margin
    if pie_chart_path:
        pdf.image(pie_chart_path, x=pdf.l_margin, y=None, w=page_width * 0.9)
        pdf.ln(5)

    pdf.set_font(pdf.active_font, 'B', 12)
    pdf.cell(0, 10, 'Bảng Trọng số chi tiết:', 0, 1, 'L')
    weights_df = pd.DataFrame(weights_dict.items(), columns=['Tiêu chí (gốc)', 'Trọng số'])
    weights_df['Tiêu chí'] = weights_df['Tiêu chí (gốc)'].apply(_nice_name)
    weights_df = weights_df[['Tiêu chí', 'Trọng số']].sort_values('Trọng số', ascending=False)
    pdf.add_df_to_table(weights_df, col_widths=[page_width * 0.7, page_width * 0.3])

    # Trang 3: Kết quả TOPSIS
    pdf.add_page()
    pdf.section_title('2. Kết quả Xếp hạng (TOPSIS)')

    # Chỉ hiển thị các cột cần thiết
    display_cols = ['Rank', 'Tên phường', 'Điểm TOPSIS (0-1)']
    final_cols = [col for col in display_cols if col in ranking_df.columns]
    df_for_table = ranking_df.sort_values('Rank')[final_cols]

    pdf.add_df_to_table(
        df_for_table,
        col_widths=[page_width * 0.15, page_width * 0.55, page_width * 0.3],
        highlight_top=True
    )

    # Trang 4: Hồ sơ Top 3 (Radar)
    pdf.add_page()
    pdf.section_title('3. Hồ sơ Top 3 Phường (Biểu đồ Radar)')
    if radar_chart_path:
        pdf.image(radar_chart_path, x=pdf.l_margin, y=None, w=page_width)
    else:
        pdf.cell(0, 10, '(Không thể tạo biểu đồ radar)', 0, 1, 'L')

    # Trang 5: Phân tích chi tiết (Bar)
    pdf.add_page()
    pdf.section_title('4. Phân tích Tiêu chí Chi tiết (Giá trị gốc)')

    if bar_chart_paths.get('population_density'):
        pdf.image(bar_chart_paths['population_density'], x=pdf.l_margin, y=None, w=page_width)
        pdf.ln(5)

    pdf.add_page()  # Thêm trang mới cho các biểu đồ tiếp

    if bar_chart_paths.get('rental_cost'):
        pdf.image(bar_chart_paths['rental_cost'], x=pdf.l_margin, y=None, w=page_width)
        pdf.ln(5)

    if bar_chart_paths.get('competition_pressure'):
        pdf.image(bar_chart_paths['competition_pressure'], x=pdf.l_margin, y=None, w=page_width)
        pdf.ln(5)

    # Trang 6: Bản đồ
    pdf.add_page()
    pdf.section_title('5. Trực quan Bản đồ Xếp hạng')
    if map_image_path:
        pdf.image(map_image_path, x=pdf.l_margin, y=None, w=page_width)
    else:
        pdf.cell(0, 10, '(Không thể tạo bản đồ)', 0, 1, 'L')

    # --- 4. Lưu và dọn dẹp ---
    final_report_name = f"Bao_cao_PDF_{safe_name}_{datetime.date.today().isoformat()}.pdf"
    final_report_path = os.path.join(REPORT_OUTPUT_DIR, final_report_name)

    try:
        pdf.output(final_report_path)
        print(f"\n--- THÀNH CÔNG! ---")
        print(f"Đã lưu báo cáo PDF tại: {final_report_path}")

        # Dọn dẹp file tạm
        temp_files = [pie_chart_path, radar_chart_path, map_image_path] + list(bar_chart_paths.values())
        for f in temp_files:
            if f and os.path.exists(f):
                os.remove(f)

        return final_report_path

    except Exception as e:
        print(f"LỖI khi lưu file PDF: {e}")
        return None