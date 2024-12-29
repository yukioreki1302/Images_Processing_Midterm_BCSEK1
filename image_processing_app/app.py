import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from utils.opencv_methods import *
from utils.manual_methods import *

# Define the application title
st.title("Image Processing Toolkit")

# Sidebar for selecting processing method and parameters
st.sidebar.header("Processing Options")

# Group similar processing methods into categories for easier navigation
method_category = st.sidebar.selectbox(
    "Choose a category:",
    ["Smoothing", "Restoration", "Segmentation", "Morphology", "Color Processing", "Compression"]
)

if method_category == "Smoothing":
    methods = ["Smoothing Linear Filter", "Median Filter", "Laplace Filter"]
elif method_category == "Restoration":
    methods = ["Sharpening", "Restore Spatial", ]
elif method_category == "Segmentation":
    methods = ["Otsu's Method", "Greylevel_thresholding's Method"]
elif method_category == "Morphology":
    methods = ["Erode", "Open"]
elif method_category == "Color Processing":
    methods = ["Greylevel Clustering", "Color Transform"]
elif method_category == "Compression":
    methods = [ "Run Length Coding"]

method = st.sidebar.selectbox("Choose a method:", methods)

# Add a checkbox to toggle between using OpenCV or manual methods
use_opencv = st.sidebar.checkbox("Use OpenCV", value=True)

# File uploader section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display warning if no image is uploaded
if uploaded_file is None:
    st.warning("Please upload an image to proceed.")
else:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display image metadata (filename and resolution)
    st.write(f"**Filename**: {uploaded_file.name}")
    height, width, _ = image.shape
    st.write(f"**Resolution**: {width} x {height} pixels")

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Create a progress bar for image processing
    progress_bar = st.progress(0)  # Initialize progress bar

    # Perform selected image processing method
    processed_image = None

    with st.spinner("Processing the image..."):
        try:
            if method == "Smoothing Linear Filter":
                processed_image = smoothing_linear_filter(image) if use_opencv else smoothing_linear_filter_manual(image)
            
            elif method == "Median Filter":
                processed_image = median_filter(image) if use_opencv else median_filter_manual(image)
                
            elif method == "Laplace Filter":
                processed_image = laplacian_filter(image) if use_opencv else laplacian_filter_manual(image)
            
            elif method == "Sharpening":
                processed_image = sharpening(image) if use_opencv else sharpening_manual(image)
            
            elif method == "Restore Spatial":
                processed_image = restore_image_spatial(image) if use_opencv else restore_image_spatial_manual(image)
            
            elif method == "Otsu's Method":
                processed_image = otsus_method(image) if use_opencv else otsus_method_manual(image)
                
            elif method == "Greylevel_thresholding's Method":
                processed_image = greylevel_thresholding(image) if use_opencv else greylevel_thresholding_manual(image)
            
            elif method == "Erode":
                processed_image = simple_morphological_erode(image) if use_opencv else simple_morphological_erode_manual(image)
                
            elif method == "Open":
                processed_image = simple_morphological_open(image) if use_opencv else simple_morphological_open_manual(image)
            
            elif method == "Greylevel Clustering":
                processed_image = greylevel_clustering(image) if use_opencv else greylevel_clustering_manual(image)
            
            elif method == "Color Transform":
                processed_image = color_transform(image) if use_opencv else color_transform_manual(image)
            
            elif method == "Run Length Coding":
                encoded = run_length_coding(image) if use_opencv else run_length_coding_manual(image)
                st.write("Run Length Coding:", encoded)

            # Update progress bar to 100% after processing
            progress_bar.progress(100)

            # Display processed image
            if processed_image is not None:
                st.image(processed_image, caption="Processed Image", use_column_width=True)

                # Convert the processed image to a BytesIO object for downloading
                is_success, buffer = cv2.imencode(".png", processed_image)
                if is_success:
                    img_bytes = buffer.tobytes()

                    # Add a download button for the processed image
                    st.download_button(
                        label="Download Processed Image",
                        data=img_bytes,
                        file_name="processed_image.png",
                        mime="image/png"
                    )
            else:
                st.error("Error processing the image. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            progress_bar.progress(0)
