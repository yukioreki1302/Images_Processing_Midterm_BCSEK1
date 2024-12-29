import cv2
import numpy as np

def smoothing_linear_filter(image):
    """Bộ lọc làm mượt tuyến tính (Linear Smoothing Filter)."""
    kernel = np.ones((5, 5), np.float32) / 25  # Kernel trung bình
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

def median_filter(image):
    """Bộ lọc trung vị (Median Filter)."""
    filtered_image = cv2.medianBlur(image, 5)
    return filtered_image

def laplacian_filter(image):
    """Bộ lọc Laplace để phát hiện biên ảnh."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh thành ảnh xám
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)  # Áp dụng Laplace
    laplacian = cv2.convertScaleAbs(laplacian)  # Chuyển lại kiểu ảnh unsigned int8
    return laplacian

def sharpening(image):
    """Làm sắc nét (Sharpening)."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Kernel làm sắc nét
    return cv2.filter2D(image, -1, kernel)

def restore_image_spatial(image):
    """Phục hồi trong trường hợp chỉ có nhiễu - lọc không gian."""
    return cv2.GaussianBlur(image, (5, 5), 0)


def otsus_method(image):
    """Phân ngưỡng tự động bằng phương pháp Otsu."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def greylevel_thresholding(image):
    threshold_value = 128  # Ngưỡng cố định
    if len(image.shape) == 3:
        gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        gray_image = image
    binary_image = np.where(gray_image >= threshold_value, 255, 0).astype(np.uint8)
    return binary_image

def simple_morphological_erode(image, operation="erode", kernel_size=5, iterations=2):
    """
    Thực hiện các phép toán hình thái học cơ bản (giãn nở, xói mòn, mở, đóng) với OpenCV,
    với kích thước kernel lớn hơn và số lần lặp tùy chỉnh.

    Parameters:
    - image: Ảnh đầu vào (numpy array)
    - operation: Phép toán hình thái học cần thực hiện, có thể là "dilate", "erode", "open", "close".
    - kernel_size: Kích thước của kernel (mặt nạ)
    - iterations: Số lần lặp lại phép toán

    Returns:
    - output: Ảnh sau khi áp dụng phép toán hình thái học.
    """
    # Tạo kernel với kích thước tùy chỉnh
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if operation == "dilate":
        # Phép giãn nở với nhiều lần lặp
        return cv2.dilate(image, kernel, iterations=iterations)
    
    elif operation == "erode":
        # Phép xói mòn với nhiều lần lặp
        return cv2.erode(image, kernel, iterations=iterations)
    
    elif operation == "open":
        # Phép mở (erosion sau đó là dilation) với nhiều lần lặp
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    elif operation == "close":
        # Phép đóng (dilation sau đó là erosion) với nhiều lần lặp
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    else:
        raise ValueError("Invalid morphological operation. Choose from 'dilate', 'erode', 'open', or 'close'.")

def simple_morphological_open(image, operation="open", kernel_size=5, iterations=2):
    """
    Thực hiện các phép toán hình thái học cơ bản (giãn nở, xói mòn, mở, đóng) với OpenCV,
    với kích thước kernel lớn hơn và số lần lặp tùy chỉnh.

    Parameters:
    - image: Ảnh đầu vào (numpy array)
    - operation: Phép toán hình thái học cần thực hiện, có thể là "dilate", "erode", "open", "close".
    - kernel_size: Kích thước của kernel (mặt nạ)
    - iterations: Số lần lặp lại phép toán

    Returns:
    - output: Ảnh sau khi áp dụng phép toán hình thái học.
    """
    # Tạo kernel với kích thước tùy chỉnh
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if operation == "dilate":
        # Phép giãn nở với nhiều lần lặp
        return cv2.dilate(image, kernel, iterations=iterations)
    
    elif operation == "erode":
        # Phép xói mòn với nhiều lần lặp
        return cv2.erode(image, kernel, iterations=iterations)
    
    elif operation == "open":
        # Phép mở (erosion sau đó là dilation) với nhiều lần lặp
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    elif operation == "close":
        # Phép đóng (dilation sau đó là erosion) với nhiều lần lặp
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    else:
        raise ValueError("Invalid morphological operation. Choose from 'dilate', 'erode', 'open', or 'close'.")


def greylevel_clustering(image):
    """Greylevel clustering."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = 2  # Số cụm
    pixel_values = gray.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(gray.shape)
    return segmented_image

def color_transform(image):
    """Phép biến đổi màu (Chuyển từ RGB sang HSV)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def run_length_coding(image):
    """Mã hóa Run Length Coding (RLC)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten()
    rle = []
    prev = flat[0]
    count = 1
    for i in range(1, len(flat)):
        if flat[i] == prev:
            count += 1
        else:
            rle.append((prev, count))
            prev = flat[i]
            count = 1
    rle.append((prev, count))
    return rle
