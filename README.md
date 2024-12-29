# Bộ Công Cụ Xử Lý Ảnh

LINK DEMO: https://image-processing-viet-bcsek1.streamlit.app/ 

CHU TUẤN VIỆT
21110112
BCSEK1- VJU

**Bộ công cụ Xử lý Ảnh** là một ứng dụng web được xây dựng với **Streamlit** để cung cấp các phương pháp xử lý ảnh cơ bản, từ làm mượt, khôi phục, phân tích đến nén ảnh. Người dùng có thể tải lên hình ảnh và chọn các phương pháp xử lý khác nhau, với tùy chọn sử dụng **OpenCV** hoặc **phương pháp thủ công**.

## Mục Lục
1. [Giới thiệu](#giới-thiệu)
2. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
3. [Cài đặt](#cài-đặt)
4. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
5. [Các phương pháp xử lý](#các-phương-pháp-xử-lý)
6. [Giới thiệu về các phương pháp](#giới-thiệu-về-các-phương-pháp)
7. [Tính năng](#tính-năng)
8. [Liên hệ](#liên-hệ)

---

## Giới thiệu

Ứng dụng này cho phép người dùng tải lên ảnh và thực hiện các phép xử lý ảnh cơ bản. Với sự kết hợp giữa **OpenCV** và các phương pháp thủ công, người dùng có thể kiểm tra và áp dụng các kỹ thuật xử lý ảnh như làm mượt, khôi phục, phân tích, hình học, và nén ảnh. Mỗi phương pháp có thể được thực hiện tự động hoặc bằng cách sử dụng mã thủ công.

---

## Yêu cầu hệ thống

- **Python 3.12**
- **Streamlit**: Để chạy ứng dụng web.
- **OpenCV**: Để xử lý ảnh.
- **NumPy**: Để xử lý mảng dữ liệu.
- **Các thư viện bổ trợ khác** như `Pillow`, , v.v.

---

## Cài đặt

Để cài đặt và chạy ứng dụng, bạn cần cài đặt các thư viện yêu cầu và chạy ứng dụng Streamlit. Dưới đây là các bước cài đặt:

1. **Clone repository**:
   ```bash
   https://github.com/yukioreki1302/Images_Processing_Midterm_BCSEK1.git
   cd image_processing_app
2. **Cài đặt các thư viện yêu cầu: Tạo môi trường ảo (khuyến khích) và cài đặt các thư viện cần thiết**:
   ```bash
   pip install -r requirements.txt
4. **Chạy ứng dụng Streamlit**:
   ```bash
   streamlit run app.py

**Hướng dẫn sử dụng**

Tải lên ảnh:

Truy cập vào giao diện của ứng dụng và tải lên một bức ảnh (JPEG, PNG hoặc JPG).

Chọn phương pháp xử lý:

Chọn danh mục phương pháp xử lý ảnh từ thanh bên (ví dụ: "Làm Mượt", "Khôi Phục", "Phân Tách", v.v.).
Chọn phương pháp cụ thể từ các lựa chọn trong mỗi danh mục.

Tùy chọn sử dụng OpenCV:

Bạn có thể chọn sử dụng OpenCV để xử lý ảnh tự động, hoặc chọn phương pháp thủ công để thực hiện các bước xử lý thủ công.

Xem kết quả:

Ảnh đã xử lý sẽ được hiển thị trực tiếp trên trang.

Sau khi xử lý xong, bạn có thể tải về ảnh đã xử lý bằng cách nhấn vào nút "Tải về ảnh đã xử lý".

![image](https://github.com/user-attachments/assets/7c66aa32-c322-4c5b-a925-abcb3cf502fa)


Ấn nút đỏ OpenCV để chuyển giữa dùng Open CV và thuật toán sử dụng toán học 

![image](https://github.com/user-attachments/assets/2dec5c31-873e-4abc-9e51-643ca060739e)
