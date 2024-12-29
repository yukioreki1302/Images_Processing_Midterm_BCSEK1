# Ứng Dụng Xử Lý Ảnh

Ứng dụng web này cho phép người dùng tải lên ảnh, chọn các phương pháp xử lý ảnh (Sử dụng OpenCV hoặc các phương pháp toán học), và xem kết quả xử lý ngay lập tức. Các phương pháp xử lý ảnh bao gồm bộ lọc làm mượt, làm sắc nét, nhận dạng đối tượng (YOLO), và các phương pháp phân tích tần số.

## Các Phương Pháp Xử Lý Ảnh

- **Bộ lọc làm mượt tuyến tính (Smoothing Linear Filter)**
- **Bộ lọc trung vị (Median Filter)**
- **Làm sắc nét ảnh (Sharpening)**
- **Nhận dạng vật thể bằng YOLO**
- **Giảm nhiễu miền tần số**
- **Lọc nghịch đảo (Inverse Filtering)**
- **Phép biến đổi Otsu**

## Yêu Cầu Cài Đặt

Ứng dụng này được xây dựng với Python và Streamlit. Bạn có thể cài đặt các thư viện yêu cầu bằng cách sử dụng tệp `requirements.txt`.

### Cài đặt các thư viện

```bash
pip install -r requirements.txt


Chạy Ứng Dụng
Sau khi cài đặt xong các thư viện, bạn có thể chạy ứng dụng bằng lệnh sau:

bash
Sao chép mã
streamlit run app.py