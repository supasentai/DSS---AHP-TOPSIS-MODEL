# report_module.py
"""
Module để tạo báo cáo PDF tổng hợp từ kết quả AHP (trọng số)
và TOPSIS (xếp hạng), bao gồm cả biểu đồ và bản đồ.

YÊU CẦU CÀI ĐẶT:
pip install fpdf2 matplotlib geopandas pyyaml pandas openpyxl
"""

import os
import datetime
import pandas as pd
import yaml
import json
import geopandas
import matplotlib.pyplot as plt
from fpdf import FPDF

# --- CẤU HÌNH ---
# Đảm bảo font hỗ trợ Unicode (Tiếng Việt)
# Bạn có thể cần tải file 'DejaVuSans.ttf' và đặt cùng thư mục
# Hoặc cài đặt gói: pip install fpdf2[fonts]
try:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    FONT_NAME = 'DejaVu'
except Exception as e:
    print(f"Cảnh báo: Không tìm thấy font 'DejaVu Sans'. "
          f"Vui lòng cài đặt font hoặc fpdf2[fonts]. Sử dụng font mặc định. Lỗi: {e}")
    plt.rcParams['font.family'] = 'Arial'
    FONT_NAME = 'Arial'

# Đường dẫn (giữ nguyên như các module khác)
# ### SỬA ĐỔI: Phân biệt default và user weights ###
DEFAULT_WEIGHTS_PATH = "data/defaultweights.yaml"
USER_WEIGHTS_PATH = "data/weights.yaml"  # File này sẽ lưu các thay đổi của người dùng
# Giữ nguyên các đường dẫn khác
METADATA_PATH = "data/metadata.json"
GEOJSON_PATH = "data/quan7_geojson.json"
RANKING_DIR = "data"
REPORT_OUTPUT_DIR = "reports"
TEMP_ASSET_DIR = os.path.join(REPORT_OUTPUT_DIR, "temp_assets")

# ### THÊM MỚI: Định nghĩa thư mục Font ###
FONT_DIR = "Font"


class PDF(FPDF):
    """
    Lớp PDF tùy chỉnh để thêm Header, Footer và các hàm tiện ích
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = "N/A"

        # ### SỬA ĐỔI: Sử dụng biến instance 'active_font' để tránh lỗi UnboundLocalError ###
        self.active_font = FONT_NAME

        try:
            # ### SỬA ĐỔI: Trỏ đường dẫn đến thư mục 'Font' ###
            font_path = os.path.join(FONT_DIR, 'DejaVuSans.ttf')
            font_path_b = os.path.join(FONT_DIR, 'DejaVuSans-Bold.ttf')
            font_path_i = os.path.join(FONT_DIR, 'DejaVuSans-Oblique.ttf')

            self.add_font(self.active_font, '', font_path, uni=True)
            self.add_font(self.active_font, 'B', font_path_b, uni=True)
            self.add_font(self.active_font, 'I', font_path_i, uni=True)

        except RuntimeError as e:
            print(f"LỖI FPDF: Không thể tải font từ thư mục '{FONT_DIR}'.")
            print(f"Đảm bảo các file .ttf (DejaVuSans,...) nằm trong thư mục '{FONT_DIR}'.")
            print(f"Lỗi chi tiết: {e}")
            print("Sử dụng font 'Arial' mặc định (có thể lỗi Tiếng Việt).")
            # ### SỬA ĐỔI: Thêm lại fallback an toàn ###
            self.active_font = 'Arial'

    def set_model_name(self, model_name):
        self.model_name = model_name

    def header(self):
        # ### SỬA ĐỔI: Phải dùng self.active_font ###
        self.set_font(self.active_font, 'B', 15)
        self.cell(0, 10, 'Báo cáo Hệ thống Hỗ trợ Quyết định (DSS)', 0, 1, 'C')
        self.set_font(self.active_font, '', 12)
        self.cell(0, 8, f'Mô hình Phân tích: {self.model_name}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        # ### SỬA ĐỔI: Phải dùng self.active_font ###
        self.set_font(self.active_font, 'I', 8)
        self.cell(0, 10, f'Trang {self.page_no()}/{{nb}}', 0, 0, 'C')
        self.set_x(-50)
        self.cell(0, 10, f'Ngày: {datetime.date.today().isoformat()}', 0, 0, 'R')

    def section_title(self, title):
        # ### SỬA ĐỔI: Phải dùng self.active_font ###
        self.set_font(self.active_font, 'B', 14)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, f' {title}', 0, 1, 'L', fill=True)
        self.ln(5)

    def add_df_to_table(self, df, col_widths=None):
        """
        Vẽ một DataFrame của pandas thành bảng trong PDF
        """
        # ### SỬA ĐỔI: Phải dùng self.active_font ###
        self.set_font(self.active_font, 'B', 10)

        # Tính toán độ rộng cột nếu không được cung cấp
        if col_widths is None:
            num_cols = len(df.columns)
            # (Page width - 2*margin) / num_cols
            equal_width = (self.w - 2 * self.l_margin) / num_cols
            col_widths = [equal_width] * num_cols

        # --- Header ---
        self.set_fill_color(220, 220, 220)
        for i, col_name in enumerate(df.columns):
            self.cell(col_widths[i], 8, str(col_name), border=1, fill=True, align='C')
        self.ln()

        # --- Body ---
        # ### SỬA ĐỔI: Phải dùng self.active_font ###
        self.set_font(self.active_font, '', 9)
        self.set_fill_color(255, 255, 255)
        for _, row in df.iterrows():
            for i, item in enumerate(row):
                # Làm tròn số float
                if isinstance(item, float):
                    item_str = f"{item:.4f}"
                else:
                    item_str = str(item)
                self.cell(col_widths[i], 7, item_str, border=1, fill=True, align='L')
            self.ln()
        self.ln(5)  # Thêm khoảng trắng sau bảng


def _nice_name(col: str) -> str:
    """Helper để làm đẹp tên cột/tiêu chí"""
    return str(col).replace("_", " ").strip().title()


def _generate_weights_pie_chart(weights_dict, output_path):
    """
    Tạo biểu đồ tròn từ dict trọng số và lưu ra file ảnh
    """
    try:
        df = pd.DataFrame(weights_dict.items(), columns=['criterion', 'weight'])
        df['criterion'] = df['criterion'].apply(_nice_name)
        df = df.sort_values('weight', ascending=False)

        # Lọc ra các trọng số > 0
        df_plot = df[df['weight'] > 0]
        if df_plot.empty:
            print("Không có trọng số > 0 để vẽ biểu đồ.")
            return None

        fig, ax = plt.subplots(figsize=(10, 7))
        wedges, texts, autotexts = ax.pie(
            df_plot['weight'],
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85
        )

        ax.axis('equal')  # Đảm bảo hình tròn

        # Thêm vòng tròn ở giữa (Donut chart)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)

        # Cấu hình legend
        ax.legend(
            wedges,
            df_plot['criterion'],
            title="Tiêu chí",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )

        plt.setp(autotexts, size=10, weight="bold", color="black")
        ax.set_title("Phân bổ Trọng số Tiêu chí (AHP)", size=16, weight="bold")

        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Đã lưu biểu đồ trọng số tại: {output_path}")
        return output_path
    except Exception as e:
        print(f"LỖI khi tạo biểu đồ trọng số: {e}")
        return None


def _generate_ranking_map(ranking_df, geojson_path, output_path):
    """
    Tạo bản đồ tĩnh GeoJSON đã tô màu theo xếp hạng và lưu ra file ảnh
    """
    try:
        gdf = geopandas.read_file(geojson_path)

        # ### SỬA ĐỔI: Tạo copy để tránh thay đổi dataframe gốc ###
        ranking_df_copy = ranking_df.copy()

        # Chuẩn hóa key để join (giống trong app.py)
        gdf['join_key'] = gdf['name'].astype(str).str.replace(" ", "")
        ranking_df_copy['join_key'] = ranking_df_copy['Tên phường'].astype(str).str.replace(" ", "")

        # Join dữ liệu không gian và dữ liệu xếp hạng
        merged_gdf = gdf.merge(ranking_df_copy, on='join_key', how='left')

        # Xử lý các phường không có dữ liệu (nếu có)
        merged_gdf['Rank'] = merged_gdf['Rank'].fillna(-1)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Vẽ bản đồ
        merged_gdf.plot(
            column='Rank',
            ax=ax,
            legend=True,
            cmap='RdYlGn_r',  # Đảo ngược: Xanh (Hạng 1) -> Đỏ (Hạng cuối)
            missing_kwds={
                "color": "lightgrey",
                "label": "Không có dữ liệu",
            },
            legend_kwds={'label': "Xếp hạng TOPSIS (1 = Tốt nhất)",
                         'orientation': "horizontal"}
        )

        # Thêm nhãn tên phường
        for x, y, label in zip(merged_gdf.geometry.centroid.x, merged_gdf.geometry.centroid.y, merged_gdf["name"]):
            ax.text(x, y, label, fontsize=8, ha='center', color='black')

        ax.set_title("Bản đồ Xếp hạng TOPSIS Quận 7", size=18, weight="bold")
        ax.axis('off')

        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Đã lưu bản đồ tại: {output_path}")
        return output_path
    except Exception as e:
        print(f"LỖI khi tạo bản đồ: {e}")
        return None


def create_full_report(model_name: str) -> str | None:
    """
    Hàm chính để tạo báo cáo PDF tổng hợp.

    Args:
        model_name: Tên của mô hình (kịch bản) đã lưu trong weights.yaml

    Returns:
        Đường dẫn đến file PDF đã tạo, hoặc None nếu thất bại.
    """
    print(f"--- Bắt đầu tạo báo cáo cho mô hình: {model_name} ---")

    # --- 0. Chuẩn bị thư mục ---
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_ASSET_DIR, exist_ok=True)

    # --- 1. Tải dữ liệu ---
    try:
        # ### SỬA ĐỔI: Tải cả default và user weights ###
        all_weights = {}

        # 1. Tải default weights (gốc)
        if os.path.exists(DEFAULT_WEIGHTS_PATH):
            with open(DEFAULT_WEIGHTS_PATH, 'r', encoding='utf-8') as f:
                all_weights.update(yaml.safe_load(f) or {})
        else:
            print(f"Cảnh báo: Không tìm thấy file {DEFAULT_WEIGHTS_PATH}. Chỉ tải file weights.yaml.")

        # 2. Tải user weights (tùy chỉnh) và ghi đè
        if os.path.exists(USER_WEIGHTS_PATH):
            with open(USER_WEIGHTS_PATH, 'r', encoding='utf-8') as f:
                all_weights.update(yaml.safe_load(f) or {})

        weights_dict = all_weights.get(model_name)
        if not weights_dict:
            print(f"LỖI: Không tìm thấy mô hình '{model_name}' trong file weights nào.")
            return None

        # Tải kết quả xếp hạng
        ranking_file = os.path.join(RANKING_DIR, f"ranking_result_{model_name}.xlsx")

        # ### SỬA ĐỔI: Thêm index_col=0 để loại bỏ cột 'Unnamed: 0' khi đọc ###
        # Chúng ta giả định cột 'Unnamed: 0' là index cũ.
        # Nếu nó không phải là index, cách lọc cột ở dưới vẫn sẽ xử lý nó.
        try:
            ranking_df = pd.read_excel(ranking_file, index_col=0)
        except ValueError:
            # Nếu file Excel không có cột index (ví dụ đã được sửa),
            # đọc lại bình thường.
            ranking_df = pd.read_excel(ranking_file)


    except FileNotFoundError as e:
        print(f"LỖI: Thiếu file dữ liệu. {e}")
        return None
    except Exception as e:
        print(f"LỖI khi tải dữ liệu: {e}")
        return None

    # --- 2. Tạo các tài sản (ảnh) ---
    print("Đang tạo tài sản trực quan...")
    pie_chart_path = _generate_weights_pie_chart(
        weights_dict,
        os.path.join(TEMP_ASSET_DIR, f"{model_name}_weights_pie.png")
    )

    map_image_path = _generate_ranking_map(
        ranking_df,
        GEOJSON_PATH,
        os.path.join(TEMP_ASSET_DIR, f"{model_name}_ranking_map.png")
    )

    # --- 3. Tạo PDF ---
    print("Đang tổng hợp file PDF...")
    pdf = PDF('P', 'mm', 'A4')  # Portrait, mm, A4
    pdf.set_model_name(model_name)
    pdf.alias_nb_pages()
    pdf.add_page()

    # Trang bìa (đơn giản)
    # ### SỬA ĐỔI: Phải dùng pdf.active_font ###
    pdf.set_font(pdf.active_font, 'B', 24)
    pdf.cell(0, 50, '', 0, 1)  # Khoảng trắng
    pdf.cell(0, 15, 'BÁO CÁO PHÂN TÍCH HỖ TRỢ QUYẾT ĐỊNH', 0, 1, 'C')
    pdf.set_font(pdf.active_font, 'B', 20)
    pdf.cell(0, 15, f'Mô hình: {model_name}', 0, 1, 'C')
    pdf.set_font(pdf.active_font, '', 14)
    pdf.cell(0, 10, f'Ngày tạo: {datetime.date.today().isoformat()}', 0, 1, 'C')
    pdf.cell(0, 10, 'Đơn vị: DSS Quận 7', 0, 1, 'C')

    # Trang 2: Trọng số AHP
    pdf.add_page()
    pdf.section_title('1. Phân bổ Trọng số (AHP)')

    if pie_chart_path and os.path.exists(pie_chart_path):
        # Chiều rộng trang A4 là 210mm, trừ 10mm lề mỗi bên = 190mm
        page_width = pdf.w - 2 * pdf.l_margin
        pdf.image(pie_chart_path, x=pdf.l_margin, y=None, w=page_width)
        pdf.ln(5)
    else:
        # ### SỬA ĐỔI: Phải dùng pdf.active_font ###
        pdf.set_font(pdf.active_font, 'I', 10)
        pdf.cell(0, 10, '(Không thể tạo biểu đồ trọng số)', 0, 1, 'L')

    # ### SỬA ĐỔI: Phải dùng pdf.active_font ###
    pdf.set_font(pdf.active_font, 'B', 12)
    pdf.cell(0, 10, 'Bảng Trọng số chi tiết:', 0, 1, 'L')
    weights_df = pd.DataFrame(weights_dict.items(), columns=['Tiêu chí (gốc)', 'Trọng số'])
    weights_df['Tiêu chí'] = weights_df['Tiêu chí (gốc)'].apply(_nice_name)
    weights_df = weights_df[['Tiêu chí', 'Trọng số']].sort_values('Trọng số', ascending=False)

    pdf.add_df_to_table(
        weights_df,
        col_widths=[(pdf.w - 2 * pdf.l_margin) * 0.7, (pdf.w - 2 * pdf.l_margin) * 0.3]
    )

    # Trang 3: Kết quả TOPSIS
    pdf.add_page()
    pdf.section_title('2. Kết quả Xếp hạng (TOPSIS)')

    # ### SỬA ĐỔI MỚI: Chỉ chọn các cột mong muốn để hiển thị ###
    # Điều này sẽ tự động loại bỏ 'Unnamed: 0' (nếu có) và 'join_key'
    display_cols = ['Tên phường', 'Điểm TOPSIS (0-1)', 'Rank']

    # Đảm bảo các cột này tồn tại trước khi lọc
    final_cols = [col for col in display_cols if col in ranking_df.columns]

    # Nếu không có cột nào hợp lệ, in dataframe gốc để gỡ lỗi
    if not final_cols:
        print("Cảnh báo: Không tìm thấy các cột hiển thị mong muốn trong ranking_df.")
        df_for_table = ranking_df
    else:
        df_for_table = ranking_df[final_cols]

    pdf.add_df_to_table(df_for_table)
    # ### KẾT THÚC SỬA ĐỔI ###

    # Trang 4: Bản đồ
    pdf.add_page()
    pdf.section_title('3. Trực quan Bản đồ Xếp hạng')

    if map_image_path and os.path.exists(map_image_path):
        page_width = pdf.w - 2 * pdf.l_margin
        img_height = (page_width * 1.0)  # Giả sử ảnh vuông (từ figsize 10,10)

        # Kiểm tra xem có đủ chỗ trên trang không
        if pdf.get_y() + img_height > pdf.page_break_trigger:
            pdf.add_page()

        pdf.image(map_image_path, x=pdf.l_margin, y=None, w=page_width)
    else:
        # ### SỬA ĐỔI: Phải dùng pdf.active_font ###
        pdf.set_font(pdf.active_font, 'I', 10)
        pdf.cell(0, 10, '(Không thể tạo bản đồ)', 0, 1, 'L')

    # --- 4. Lưu và dọn dẹp ---
    final_report_name = f"Report_{model_name.replace(' ', '_')}_{datetime.date.today().isoformat()}.pdf"
    final_report_path = os.path.join(REPORT_OUTPUT_DIR, final_report_name)

    try:
        pdf.output(final_report_path)
        print(f"\n--- THÀNH CÔNG! ---")
        print(f"Đã lưu báo cáo tại: {final_report_path}")

        # Dọn dẹp file tạm
        if pie_chart_path and os.path.exists(pie_chart_path):
            os.remove(pie_chart_path)
        if map_image_path and os.path.exists(map_image_path):
            os.remove(map_image_path)

        return final_report_path

    except Exception as e:
        print(f"LỖI khi lưu file PDF: {e}")
        return None


# --- Để KIỂM TRA (TEST) ---
# Bạn có thể chạy file này trực tiếp để kiểm tra 1 mô hình
if __name__ == "__main__":
    # GIẢ SỬ bạn có một mô hình tên là 'Nang_cao_Chat_luong_Cuoc_song'
    # và file 'data/ranking_result_Nang_cao_Chat_luong_Cuoc_song.xlsx' đã tồn tại

    # Hãy thay 'TEN_MO_HINH_CUA_BAN' bằng một tên mô hình
    # có thật trong file weights.yaml của bạn.
    TEST_MODEL_NAME = "TEST_MODEL_NAME"  # <--- THAY ĐỔI TÊN NÀY

    print("Chạy kiểm tra module báo cáo...")

    # Tạo dữ liệu giả nếu cần
    if not os.path.exists(DEFAULT_WEIGHTS_PATH):
        print(f"Tạo file {DEFAULT_WEIGHTS_PATH} giả để test...")
        os.makedirs("data", exist_ok=True)
        demo_weights = {
            TEST_MODEL_NAME: {
                'so_dan': 0.2, 'mat_do_dan_so': 0.1, 'ty_le_tang_dan_so': 0.1,
                'so_ho_ngheo': 0.3, 'so_truong_mam_non': 0.15, 'so_truong_tieu_hoc': 0.15
            }
        }
        with open(DEFAULT_WEIGHTS_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(demo_weights, f)

    if not os.path.exists(f"data/ranking_result_{TEST_MODEL_NAME}.xlsx"):
        print(f"Tạo file ranking_result_{TEST_MODEL_NAME}.xlsx giả để test...")
        demo_ranking = pd.DataFrame({
            'Tên phường': ['Phường Tân Thuận Đông', 'Phường Tân Thuận Tây', 'Phường Bình Thuận'],
            'Điểm TOPSIS (0-1)': [0.75, 0.62, 0.45],
            'Rank': [1, 2, 3]
        })
        demo_ranking.to_excel(f"data/ranking_result_{TEST_MODEL_NAME}.xlsx", index=False)  # Đã thêm index=False

    if not os.path.exists(GEOJSON_PATH):
        print(f"LỖI: Không tìm thấy {GEOJSON_PATH}. Cần file này để test bản đồ.")
    else:
        create_full_report(TEST_MODEL_NAME)

