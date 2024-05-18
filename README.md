# Image Processing Application

## Overview

This Image Processing Application allows users to apply various filters and transformations to grayscale images. It is built using Streamlit, a popular Python framework for creating interactive web applications. Users can upload a subject image, choose from several filters, and view the results directly in the application interface.

## Features

- **Upload Subject Image**: Users can upload their own images for processing.
- **Initial Filter Selection**: Choose from Lowpass Gaussian Filter, Lowpass Butterworth Filter, Highpass Laplacian Filter, and Histogram Matching.
- **Additional Processing Options**: Apply noise, box filter, Gaussian filter, median filter, and gamma correction to the processed image.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.6 or higher
- Streamlit
- OpenCV
- NumPy
- Pillow

You can install the required packages using pip:

```bash
pip install streamlit opencv-python-headless numpy pillow
```

### Running the Application

Clone the repository and navigate to the directory:

```bash
git clone <repository-url>
cd <repository-directory>
```

Run the Streamlit application:

```bash
streamlit run app.py
```

### Code Overview

#### Import Libraries

```python
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
```

#### Streamlit Page Configuration

```python
st.set_page_config(page_title="Image Processing Application", layout="wide")
```

#### Utility Functions

- **Resize Image**: Resize images to fit within specified dimensions.

```python
def resize_image(image, max_width=400, max_height=400):
    height, width = image.shape[:2]
    if height > max_height or width > max_width:
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    return image
```

- **Display Image**: Display images with captions.

```python
def display_image(image, caption, width=None):
    st.image(image, caption=caption, use_column_width=(width is None), width=width)
```

- **Print Stats**: Print mean and standard deviation of the image.

```python
def print_stats(image):
    if image is not None:
        mean_val = np.mean(image)
        std_dev = np.std(image)
        st.write(f"Mean: {mean_val:.2f}, Standard Deviation: {std_dev:.2f}")
```

- **Load Image**: Load an image from a given path.

```python
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        st.error(f"Unable to load image at {image_path}.")
        return None
    return resize_image(image)
```

#### Filter Functions

- **Salt & Pepper Noise**: Add salt and pepper noise to the image.

```python
def salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy_image = image.copy()
    salt_mask = np.random.rand(*image.shape) < salt_prob
    pepper_mask = np.random.rand(*image.shape) < pepper_prob
    noisy_image[salt_mask] = 255
    noisy_image[pepper_mask] = 0
    return noisy_image
```

- **Box Filter**: Apply a box filter to the image.

```python
def box_filter(image):
    return cv2.blur(image, (5, 5))
```

- **Gaussian Filter**: Apply a Gaussian filter to the image.

```python
def gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)
```

- **Median Filter**: Apply a median filter to the image.

```python
def median_filter(image):
    return cv2.medianBlur(image, 5)
```

- **Gamma Correction**: Apply gamma correction to the image.

```python
def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
```

- **Lowpass Gaussian Filter**: Apply a lowpass Gaussian filter to the image.

```python
def lowpass_gaussian_filter(image):
    return cv2.GaussianBlur(image, (15, 15), 0)
```

- **Lowpass Butterworth Filter**: Apply a lowpass Butterworth filter to the image.

```python
def lowpass_butterworth_filter(image, d0=30, n=2):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    D = np.sqrt((u - ccol)**2 + (v - crow)**2)
    H = 1 / (1 + (D / d0)**(2 * n))

    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    dft_shift[:, :, 0] *= H
    dft_shift[:, :, 1] *= H
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return np.uint8(img_back)
```

- **Highpass Laplacian Filter**: Apply a highpass Laplacian filter to the image.

```python
def highpass_laplacian_filter(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    return laplacian
```

- **Histogram Matching**: Match the histogram of the source image to the reference image.

```python
def histogram_matching(source, reference):
    src_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
    
    src_cdf = src_hist.cumsum()
    ref_cdf = ref_hist.cumsum()
    
    src_cdf_normalized = src_cdf * ref_hist.max() / src_cdf.max()
    ref_cdf_normalized = ref_cdf * src_hist.max() / ref_cdf.max()
    
    lookup_table = np.zeros(256, dtype=np.uint8)
    ref_cdf_min = ref_cdf[ref_cdf > 0].min()
    for i in range(256):
        closest_index = np.abs(ref_cdf_normalized - src_cdf_normalized[i]).argmin()
        lookup_table[i] = closest_index
    
    matched_image = cv2.LUT(source, lookup_table)
    return matched_image
```

#### Main Application

- **Application Title**: Set the main title of the application.

```python
st.title("Image Processing Application")
```

- **Sidebar for Initial Filter Selection**: Allow users to select an initial filter from the sidebar.

```python
image_paths = {
    "Lowpass Gaussian Filter (Spatial Domain)": 'E:\\CV Assignment\\Gussain Low Pas.jpeg',
    "Lowpass Butterworth Filter (Frequency Domain)": 'E:\\CV Assignment\\Butterworth filter.jpg',
    "Highpass Laplacian Filter (Spatial Domain)": 'E:\\CV Assignment\\Laplacian.jpeg',
    "Histogram Matching": 'E:\\CV Assignment\\Histogram.jpeg'
}

st.sidebar.title("Select an Initial Filter")
filter_option = st.sidebar.selectbox("Choose a filter", list(image_paths.keys()))
```

- **Load and Display Original Image**: Load and display the original image based on the selected filter.

```python
if filter_option in image_paths:
    original_image = load_image(image_paths[filter_option])
    if original_image is not None:
        display_image(original_image, "Original Image")
        print_stats(original_image)
        
        if 'processed_image' not in st.session_state:
            st.session_state.processed_image = original_image
```

- **Apply Selected Filter**: Apply the selected filter to the original image and display the result.

```python
        if filter_option == "Lowpass Gaussian Filter (Spatial Domain)":
            st.header("Lowpass Gaussian Filter (Spatial Domain)")
            st.session_state.processed_image = lowpass_gaussian_filter(original_image)
            display_image(st.session_state.processed_image, "Lowpass Gaussian Filtered Image")
        
        elif filter_option == "Lowpass Butterworth Filter (Frequency Domain)":
            st.header("Lowpass Butterworth Filter (Frequency Domain)")
            st.session_state.processed_image = lowpass_butterworth_filter(original_image)
            display_image(st.session_state.processed_image, "Lowpass Butterworth Filtered Image")
        
        elif filter_option == "Highpass Laplacian Filter (Spatial Domain)":
            st.header("Highpass Laplacian Filter (Spatial Domain)")
            st.session_state.processed_image = highpass_laplacian_filter(original_image)
            display_image(st.session_state.processed_image, "Highpass Laplacian Filtered Image")
        
        elif filter_option == "Histogram Matching":
            st.header("Histogram Matching")
            uploaded_reference = st.file_uploader("Choose a reference image...", type=["jpg

", "jpeg", "png"])
            if uploaded_reference is not None:
                file_bytes_ref = np.asarray(bytearray(uploaded_reference.read()), dtype=np.uint8)
                reference_image = cv2.imdecode(file_bytes_ref, cv2.IMREAD_GRAYSCALE)
                reference_image = resize_image(reference_image)
                st.session_state.processed_image = histogram_matching(original_image, reference_image)
                display_image(reference_image, "Reference Image")
                display_image(st.session_state.processed_image, "Histogram Matched Image")
```

- **Sidebar for Additional Processing Options**: Allow users to apply additional processing options from the sidebar.

```python
        st.sidebar.title("Image Processing Options")
        
        if 'processed_images' not in st.session_state:
            st.session_state.processed_images = {
                "Noisy Image": None,
                "Box Filtered Image": None,
                "Gaussian Filtered Image": None,
                "Median Filtered Image": None,
                "Gamma Corrected Image": None
            }
        
        with st.sidebar.expander("Add Noise"):
            if st.button("Add Salt & Pepper Noise"):
                noisy_image = salt_pepper_noise(st.session_state.processed_image)
                st.session_state.processed_images["Noisy Image"] = noisy_image
        
        with st.sidebar.expander("Filters"):
            if st.button("Apply Box Filter"):
                filtered_image = box_filter(st.session_state.processed_image)
                st.session_state.processed_images["Box Filtered Image"] = filtered_image

            if st.button("Apply Gaussian Filter"):
                filtered_image = gaussian_filter(st.session_state.processed_image)
                st.session_state.processed_images["Gaussian Filtered Image"] = filtered_image

            if st.button("Apply Median Filter"):
                filtered_image = median_filter(st.session_state.processed_image)
                st.session_state.processed_images["Median Filtered Image"] = filtered_image
        
        with st.sidebar.expander("Gamma Correction"):
            gamma_value = st.slider("Gamma Correction", 0.1, 3.0, 1.0)
            if st.button("Apply Gamma Correction"):
                gamma_corrected = gamma_correction(st.session_state.processed_image, gamma_value)
                st.session_state.processed_images["Gamma Corrected Image"] = gamma_corrected
```

- **Display Processed Images**: Display processed images in the main interface and allow users to select and view a larger version.

```python
        st.markdown("## Processed Images")
        cols = st.columns(5)
        captions = ["Noisy Image", "Box Filtered Image", "Gaussian Filtered Image", "Median Filtered Image", "Gamma Corrected Image"]
        for i, caption in enumerate(captions):
            if st.session_state.processed_images[caption] is not None:
                cols[i].image(st.session_state.processed_images[caption], caption=caption, width=100)
        
        selected_caption = st.selectbox("Select an image to view larger", captions)
        if st.session_state.processed_images[selected_caption] is not None:
            st.image(st.session_state.processed_images[selected_caption], caption=f"Large view: {selected_caption}", use_column_width=True)
```

## Conclusion

This application provides a user-friendly interface for applying various image processing techniques to grayscale images. By leveraging Streamlit, it allows for interactive and real-time visualization of the effects of different filters and transformations. Users can upload their images, select filters, and further process the images with additional options, making it a versatile tool for image processing tasks.
