import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
f = cv2.imread('C:\\Users\\gksac\\OneDrive\\Desktop\\linesDetected.png')

# Resize the image for display using cv2.imshow()
resize_factor = 0.5  # Adjust this factor as needed
f_resized = cv2.resize(f, None, fx=resize_factor, fy=resize_factor)

# Display the original image using cv2.imshow()
cv2.imshow('Original Image', f_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the image to grayscale
f_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

# Transform the grayscale image into the frequency domain, f_gray --> F_gray
F_gray = np.fft.fft2(f_gray)
Fshift_gray = np.fft.fftshift(F_gray)

# Display the magnitude spectrum of the grayscale image using Matplotlib
plt.figure(figsize=(5, 5))
plt.imshow(np.log1p(np.abs(F_gray)), cmap='gray')
plt.axis('off')
plt.show()

# Display the shifted magnitude spectrum of the grayscale image using Matplotlib
plt.figure(figsize=(5, 5))
plt.imshow(np.log1p(np.abs(Fshift_gray)), cmap='gray')
plt.axis('off')
plt.show()

# Create Gaussian Filter: Low Pass Filter
M, N = f_gray.shape
H = np.zeros((M, N), dtype=np.float32)
D0 = 10
for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
        H[u, v] = np.exp(-D ** 2 / (2 * D0 * D0))

# Resize the filter for display
H_resized = cv2.resize(H, f_resized.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)

# Display the Gaussian filter using Matplotlib
plt.figure(figsize=(5, 5))
plt.imshow(H_resized, cmap='gray')
plt.axis('off')
plt.show()

# Apply the filter in the frequency domain
Gshift_gray = Fshift_gray * H
G_gray = np.fft.ifftshift(Gshift_gray)
g_gray = np.abs(np.fft.ifft2(G_gray))

# Convert the filtered image to uint8 format
g_uint8 = cv2.normalize(g_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Display the filtered image using cv2.imshow()
cv2.imshow('Filtered Image', g_uint8)
cv2.waitKey(0)
cv2.destroyAllWindows()
