# 📊 Hệ thống Hỗ trợ Ra quyết định (DSS): Lựa chọn Địa điểm Kinh doanh tại Quận 7, TP.HCM

Đây là đồ án môn học "Hệ thống Hỗ trợ Ra quyết định (DSS)" tại Đại học Kinh tế TP. Hồ Chí Minh (UEH).

Dự án này xây dựng một hệ thống DSS hoàn chỉnh để hỗ trợ các doanh nghiệp đưa ra quyết định lựa chọn địa điểm mở chi nhánh mới (văn phòng, kho xưởng, nhà máy) tại 10 phường thuộc Quận 7, TP.HCM.

**Tác giả:**
* Nguyễn Đình Lương
* Trần Quốc Bảo
* Nguyễn Huỳnh Tấn Phát
* Hồ Đức Nhân Thiện
* Nguyễn Ngọc Minh Đức

**Giảng viên hướng dẫn:** TS. Nguyễn Thành Huy

---

## 🎯 Vấn đề
Việc lựa chọn một địa điểm kinh doanh mới là một bài toán phức tạp, mang tính bán cấu trúc (semi-structured) và đa tiêu chí (MCDM). Người ra quyết định phải cân nhắc nhiều yếu tố mâu thuẫn nhau (ví dụ: giá thuê rẻ so với mật độ dân cư cao).

Hệ thống này được xây dựng để giải quyết bài toán này bằng cách định lượng hóa các tiêu chí và xếp hạng các phương án (các phường) một cách khoa học.

## 📈 Phương pháp luận
Hệ thống kết hợp hai phương pháp MCDM phổ biến:

1.  **AHP (Analytic Hierarchy Process):** Dùng để xác định trọng số (mức độ quan trọng) của các tiêu chí. Hệ thống cho phép người dùng tự định nghĩa trọng số thông qua ma trận so sánh cặp hoặc sử dụng các kịch bản có sẵn.
2.  **TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution):** Dùng để xếp hạng 10 phương án (phường) dựa trên bộ trọng số AHP đã cung cấp và dữ liệu thực tế của các phường.

## ✨ Tính năng chính
* **Phân tích theo Kịch bản:** Cung cấp 3 kịch bản cài đặt sẵn với các bộ trọng số AHP khác nhau:
    * **🏢 Office (Mở văn phòng):** Ưu tiên Chất lượng hạ tầng, An ninh, Tiện ích.
    * **📦 Warehouse (Thuê kho xưởng):** Ưu tiên Chi phí thuê, Khoảng cách tiếp cận (logistics).
    * **🏭 Factory (Xây nhà máy):** Ưu tiên Chi phí thuê, An toàn, Tiềm năng tăng trưởng (GDP).
* **Tùy chỉnh Trọng số (AHP):** Giao diện cho phép người dùng tự điều chỉnh trọng số của 9 tiêu chí thông qua thanh trượt hoặc nhập ma trận so sánh cặp.
* **Xếp hạng (TOPSIS):** Tự động tính toán và hiển thị bảng xếp hạng 10 phường dựa trên kịch bản được chọn.
* **Phân tích Độ nhạy (What-if):** Cho phép người dùng thay đổi nhẹ các trọng số để xem kết quả xếp hạng có thay đổi đáng kể hay không.
* **Trực quan hóa Dữ liệu:** Hiển thị kết quả dưới nhiều hình thức:
    * Bảng xếp hạng chi tiết.
    * Biểu đồ Radar (so sánh Top 3).
    * **Bản đồ nhiệt (Heatmap)** trực quan trên bản đồ Quận 7.

## 🛠️ Công nghệ sử dụng
* **Ngôn ngữ:** Python
* **Giao diện:** Streamlit
* **Thư viện (chính):** Pandas, NumPy, PyYAML (để quản lý file trọng số `weights.yaml`)
* **Giao diện thay thế:** Hệ thống cũng có một phiên bản CLI (Console Line Interface) để kiểm thử và chạy nhanh.

---

## 🚀 Cài đặt & Khởi chạy

Bạn cần có **Python 3.8+** và **Git** đã được cài đặt.

**1. Clone Repository**
```bash
git clone [https://github.com/supasentai/DSS---AHP-TOPSIS-MODEL.git](https://github.com/supasentai/DSS---AHP-TOPSIS-MODEL.git)
cd DSS---AHP-TOPSIS-MODEL
