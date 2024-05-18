import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

def ButterworthFilter(f):
    
    # transform image into freq. domain and shifted
    F = np.fft.fft2(f)
    Fshift = np.fft.fftshift(F)

    plt.imshow(np.log1p(np.abs(Fshift)), cmap='gray')
    plt.axis('off')
    plt.show()

    # Butterwort Low Pass Filter
    M,N = f.shape
    

    # Butterworth High Pass Filter
    HPF = np.zeros((M,N), dtype=np.float32)
    D0 = 10
    n = 1
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            HPF[u,v] = 1 / (1 + (D0/D)**n)
        
    plt.imshow(HPF, cmap='gray')
    plt.axis('off')
    plt.show()

    # frequency domain image filters
    Gshift = Fshift * HPF
    G = np.fft.ifftshift(Gshift)
    g = np.abs(np.fft.ifft2(G))

    plt.imshow(g, cmap='gray')
    plt.axis('off')
    plt.show()
    # Convert the filtered image to uint8 format
    g_uint8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    
    return g_uint8
    

def GaussianFilter(f_gray):
    # Convert the image to grayscale
    

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

    # Display the Gaussian filter using Matplotlib
    plt.figure(figsize=(5, 5))
    plt.imshow(H, cmap='gray')
    plt.axis('off')
    plt.show()

    # Apply the filter in the frequency domain
    Gshift_gray = Fshift_gray * H
    G_gray = np.fft.ifftshift(Gshift_gray)
    g_gray = np.abs(np.fft.ifft2(G_gray))

    # Convert the filtered image to uint8 format
    g_uint8 = cv2.normalize(g_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    
    return g_uint8


def process_image():
    global selected_radio
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load the image
        image = cv2.imread(file_path,0)
        
        if selected_radio == 1:
            #gussain filter low pass filter
            imageFilter=GaussianFilter(image)
            print("Option 2 selected")
        # Add your logic for option 1 here
        elif selected_radio == 2:
            imageFilter= ButterworthFilter(image)
            print("Option 2 selected")
        # Add your logic for option 2 here
        elif selected_radio == 3:
            print("Option 3 selected")
        # Add your logic for option 3 here
        elif selected_radio == 4:
            print("Option 4 selected")
        
       
        
        
        showImage(image)
        showFilterImage(imageFilter)
        
        
       
        
        
def showFilterImage(image1):
    # Convert the processed image to PIL format for displaying in Tkinter
    image = Image.fromarray(image1)

        # Convert PIL image to Tkinter PhotoImage
    photo = ImageTk.PhotoImage(image)
    

    # Resize the processed image to fit within the window frame
    image = image.resize((100, 50), Image.ANTIALIAS)


        # Display the processed image in the Tkinter application
    image_label.config(image=photo)
    image_label.image = photo

        
def showImage(image_gray):
    # Resize the processed image to fit within a fixed size area
    max_width = 400
    max_height = 300
    height, width = image_gray.shape
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        width = int(width * scale)
        height = int(height * scale)
        image_gray = cv2.resize(image_gray, (width, height))

  # Convert the grayscale image to RGB format for display
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    image_rgb = Image.fromarray(image_rgb)
    image_rgb = ImageTk.PhotoImage(image_rgb)

   # Update the label to display the processed image
    processed_image_label.config(image=image_rgb)
    processed_image_label.image = image_rgb        

# Create the main application window
app = tk.Tk()
app.title("Image Processing Application")

# Set the window size and position it in the center of the screen
window_width = 600
window_height = 400
screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2
app.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
# Add a label for the application title
title_label = tk.Label(app, text="Image Processing Application", font=("Helvetica", 20), pady=10)
title_label.pack()


selected_radio=0

def function1():
    global selected_radio
    selected_radio=1

def function2():
    global selected_radio
    
    selected_radio=2

def function3():
    global selected_radio
    selected_radio=3

def function4():
    global selected_radio
    selected_radio=4
# Add radio buttons
radio_var = tk.IntVar()


radio1 = tk.Radiobutton(app, text="Gaussian Filter: Low Pass Filter", variable=radio_var, value=1, command=function1)
radio1.pack()
radio1.configure(bg="#e1e1e1", fg="#333", font=("Helvetica", 12), padx=10, pady=5)

radio2 = tk.Radiobutton(app, text=" Butterworth High Pass Filter", variable=radio_var, value=2, command=function2)
radio2.pack()
radio2.configure(bg="#e1e1e1", fg="#333", font=("Helvetica", 12), padx=10, pady=5)
radio3 = tk.Radiobutton(app, text="Laplacian", variable=radio_var, value=3, command=function3)
radio3.pack()
radio3.configure(bg="#e1e1e1", fg="#333", font=("Helvetica", 12), padx=10, pady=5)

radio4 = tk.Radiobutton(app, text="Histogram Matching", variable=radio_var, value=4, command=function4)
radio4.pack()
radio4.configure(bg="#e1e1e1", fg="#333", font=("Helvetica", 12), padx=10, pady=5)

# Add a button to select and process the image
process_button = tk.Button(app, text="Select and Process Image", font=("Helvetica", 14), command=process_image, padx=20, pady=10, bg="#4CAF50", fg="white")
process_button.pack()


# Add a label to display the processed image
processed_image_label = tk.Label(app, text="Original Image")
processed_image_label.pack(side="left", padx=10)

# Add a label to display the processed image
image_label = tk.Label(app, text="Gussain FIlter Image")
image_label.pack(side="right", padx=10)

# Run the application loop
app.mainloop()
