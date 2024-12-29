import numpy as np

def smoothing_linear_filter_manual(image):
    """Bộ lọc làm mượt tuyến tính (Linear Smoothing Filter) không dùng OpenCV."""
    kernel = np.ones((5, 5)) / 25  # Kernel trung bình
    padded_image = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode='constant')
    output = np.zeros_like(image)
    
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for c in range(image.shape[2]):
                output[x, y, c] = np.sum(kernel * padded_image[x:x+5, y:y+5, c])
    return output

def median_filter_manual(image):
    """Bộ lọc trung vị (Median Filter) không dùng OpenCV."""
    padded_image = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode='constant')
    output = np.zeros_like(image)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for c in range(image.shape[2]):
                window = padded_image[x:x+5, y:y+5, c].flatten()
                output[x, y, c] = np.median(window)
    return output

def laplacian_filter_manual(image):
    """Bộ lọc Laplace để phát hiện biên ảnh mà không sử dụng OpenCV."""
    # Chuyển ảnh sang ảnh xám nếu ảnh đầu vào là ảnh màu
    if len(image.shape) == 3:
        gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Chuyển đổi RGB sang grayscale
    else:
        gray_image = image  # Nếu ảnh đã là ảnh xám

    # Định nghĩa kernel Laplace
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    # Lấy chiều cao và chiều rộng của ảnh
    height, width = gray_image.shape

    # Tạo một ảnh kết quả (cùng kích thước ảnh gốc)
    output_image = np.zeros_like(gray_image)

    # Áp dụng phép toán tích chập (convolution)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Lấy vùng con của ảnh xung quanh pixel (i, j)
            region = gray_image[i-1:i+2, j-1:j+2]
            # Tính toán Laplacian bằng phép nhân với kernel
            output_image[i, j] = np.sum(region * kernel)

    # Chuyển đổi ảnh kết quả về kiểu uint8 và clip lại trong phạm vi [0, 255]
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    return output_image

def sharpening_manual(image):
    """Làm sắc nét (Sharpening) không dùng OpenCV."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='constant')
    output = np.zeros_like(image)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for c in range(image.shape[2]):
                output[x, y, c] = np.sum(kernel * padded_image[x:x+3, y:y+3, c])
    return np.clip(output, 0, 255)

def restore_image_spatial_manual(image):
    """Phục hồi ảnh trong trường hợp chỉ có nhiễu - lọc không gian."""
    kernel = np.ones((5, 5)) / 25
    padded_image = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode='constant')
    output = np.zeros_like(image)

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for c in range(image.shape[2]):
                output[x, y, c] = np.sum(kernel * padded_image[x:x+5, y:y+5, c])
    return output


def otsus_method_manual(image):
    """Phân ngưỡng Otsu không dùng OpenCV."""
    gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
    hist, bins = np.histogram(gray.ravel(), bins=256, range=(0, 256))

    total = gray.size
    current_max, threshold = 0, 0
    sum_total, sum_foreground, weight_background = 0, 0, 0
    for t in range(256):
        sum_total += t * hist[t]
    
    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_foreground += t * hist[t]
        mean_background = sum_foreground / weight_background
        mean_foreground = (sum_total - sum_foreground) / weight_foreground

        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = t

    binary_image = (gray > threshold).astype(np.uint8) * 255
    return binary_image

def greylevel_thresholding_manual(image):
    """
    Apply Greylevel Thresholding to an image using a fixed threshold value (128).

    Parameters:
    - image: Input image (numpy array).

    Returns:
    - Binary image: A binary image where pixels are set to 255 if above the fixed threshold (128),
                     or 0 if below the threshold.
    """
    # Set a fixed threshold value (128)
    threshold_value = 128

    # Convert to grayscale if image is not already in grayscale
    if len(image.shape) == 3:
        # Calculate the grayscale image using a weighted sum of RGB values
        gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        gray_image = image
    
    # Apply the thresholding (element-wise operation)
    binary_image = np.where(gray_image >= threshold_value, 255, 0).astype(np.uint8)
    
    return binary_image

def simple_morphological_erode_manual(image, operation="erode", kernel_size=5, iterations=2):
    """
    Thực hiện các phép toán hình thái học cơ bản (giãn nở, xói mòn, mở, đóng) mà không sử dụng OpenCV,
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
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2), (0, 0)), mode='constant')  # Padding
    output = np.copy(image)

    # Lặp qua từng kênh màu (đối với ảnh màu)
    for channel in range(image.shape[2]):  # Duyệt qua từng kênh R, G, B
        for _ in range(iterations):
            temp_output = np.copy(output)
            for x in range(kernel_size//2, padded_image.shape[0] - kernel_size//2):
                for y in range(kernel_size//2, padded_image.shape[1] - kernel_size//2):
                    window = padded_image[x-kernel_size//2:x+kernel_size//2+1, y-kernel_size//2:y+kernel_size//2+1, channel]
                    if operation == "dilate":
                        temp_output[x-kernel_size//2, y-kernel_size//2, channel] = np.max(window * kernel)
                    elif operation == "erode":
                        temp_output[x-kernel_size//2, y-kernel_size//2, channel] = np.min(window * kernel)
                    elif operation == "open":
                        erosion = np.min(window * kernel)
                        temp_output[x-kernel_size//2, y-kernel_size//2, channel] = np.max(window * kernel) if erosion > 0 else 0
                    elif operation == "close":
                        dilation = np.max(window * kernel)
                        temp_output[x-kernel_size//2, y-kernel_size//2, channel] = np.min(window * kernel) if dilation > 0 else 0
            output = np.copy(temp_output)
    
    return output

def simple_morphological_open_manual(image, operation="open", kernel_size=5, iterations=2):
    """
    Thực hiện các phép toán hình thái học cơ bản (giãn nở, xói mòn, mở, đóng) mà không sử dụng OpenCV,
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
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2), (0, 0)), mode='constant')  # Padding
    output = np.copy(image)

    # Lặp qua từng kênh màu (đối với ảnh màu)
    for channel in range(image.shape[2]):  # Duyệt qua từng kênh R, G, B
        for _ in range(iterations):
            temp_output = np.copy(output)
            for x in range(kernel_size//2, padded_image.shape[0] - kernel_size//2):
                for y in range(kernel_size//2, padded_image.shape[1] - kernel_size//2):
                    window = padded_image[x-kernel_size//2:x+kernel_size//2+1, y-kernel_size//2:y+kernel_size//2+1, channel]
                    if operation == "dilate":
                        temp_output[x-kernel_size//2, y-kernel_size//2, channel] = np.max(window * kernel)
                    elif operation == "erode":
                        temp_output[x-kernel_size//2, y-kernel_size//2, channel] = np.min(window * kernel)
                    elif operation == "open":
                        erosion = np.min(window * kernel)
                        temp_output[x-kernel_size//2, y-kernel_size//2, channel] = np.max(window * kernel) if erosion > 0 else 0
                    elif operation == "close":
                        dilation = np.max(window * kernel)
                        temp_output[x-kernel_size//2, y-kernel_size//2, channel] = np.min(window * kernel) if dilation > 0 else 0
            output = np.copy(temp_output)
    
    return output

def greylevel_clustering_manual(image, k=2):
    """Greylevel clustering không dùng OpenCV."""
    gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    pixel_values = gray.flatten()

    centroids = np.linspace(0, 255, k)
    for _ in range(10):  # Iterative clustering
        distances = np.abs(pixel_values[:, None] - centroids)
        labels = np.argmin(distances, axis=1)
        for i in range(k):
            if np.any(labels == i):
                centroids[i] = np.mean(pixel_values[labels == i])

    clustered = centroids[labels].reshape(gray.shape)
    return clustered.astype(np.uint8)

def color_transform_manual(image, transform_type="hsv"):
    """Phép biến đổi màu cơ bản không dùng OpenCV."""
    
    # Chuyển đổi thành ảnh xám (grayscale)
    if transform_type == "grayscale":
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        return gray.astype(np.uint8)
    
    # Đảo ngược các giá trị màu trong ảnh (màu đối xứng)
    elif transform_type == "invert":
        return np.clip(255 - image, 0, 255).astype(np.uint8)
    
    # Chuyển đổi từ RGB sang HSV theo cách OpenCV
    elif transform_type == "hsv":
        image = image.astype(np.float32) / 255.0  # Chuyển sang phạm vi [0, 1]
        R, G, B = image[..., 0], image[..., 1], image[..., 2]
        
        # Tính toán giá trị Max và Min
        C_max = np.max(image, axis=-1)
        C_min = np.min(image, axis=-1)
        Delta = C_max - C_min
        
        # Hue (H) calculation
        H = np.zeros_like(C_max)
        mask = Delta != 0
        H[mask] = np.where(R[mask] == C_max[mask], (G[mask] - B[mask]) / Delta[mask], H[mask])
        H[mask] = np.where(G[mask] == C_max[mask], (B[mask] - R[mask]) / Delta[mask] + 2, H[mask])
        H[mask] = np.where(B[mask] == C_max[mask], (R[mask] - G[mask]) / Delta[mask] + 4, H[mask])
        H = (H / 6) % 1  # Đảm bảo Hue nằm trong khoảng [0, 1]
        H = H * 180  # Chuyển H sang phạm vi [0, 180] thay vì [0, 360]

        # Saturation (S) calculation
        S = np.zeros_like(C_max)
        S[mask] = Delta[mask] / C_max[mask]
        
        # Value (V) calculation
        V = C_max
        
        # Chuyển HSV sang kiểu dữ liệu [0, 255]
        hsv_image = np.stack((H, S * 255, V * 255), axis=-1)
        return np.clip(hsv_image, 0, 255).astype(np.uint8)  # Đảm bảo giá trị nằm trong phạm vi [0, 255]

    # Chuyển đổi từ RGB sang CMYK
    elif transform_type == "cmyk":
        image = image.astype(np.float32) / 255.0  # Chuyển sang phạm vi [0, 1]
        R, G, B = image[..., 0], image[..., 1], image[..., 2]
        
        K = 1 - np.max(image, axis=-1)  # Black component
        
        # Tránh chia cho 0 khi K = 1
        mask_K_1 = K == 1
        C = (1 - R - K) / (1 - K)
        M = (1 - G - K) / (1 - K)
        Y = (1 - B - K) / (1 - K)
        
        # Với K = 1, set C, M, Y thành 0
        C[mask_K_1] = 0
        M[mask_K_1] = 0
        Y[mask_K_1] = 0
        
        # Tránh NaN
        C[np.isnan(C)] = 0
        M[np.isnan(M)] = 0
        Y[np.isnan(Y)] = 0
        
        cmyk_image = np.stack((C, M, Y, K), axis=-1)
        return np.clip(cmyk_image, 0, 1)  # Đảm bảo giá trị nằm trong phạm vi [0, 1]

    else:
        raise ValueError("Unsupported transform type")


def run_length_coding_manual(image):
    """Mã hóa Run Length Coding (RLC) không dùng OpenCV."""
    gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
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

