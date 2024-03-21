import os
import cv2
import pandas as pd

folder_path = r'C:\Users\Tariq\Desktop\Tariq\PNW\Spring 2024\ITS 365\KNN\archive\train\nascar racing'

# Check if the folder exists
if not os.path.exists(folder_path):
    print(f"Error: Folder '{folder_path}' does not exist.")
    exit()

# Create lists to store condition values
image_names = []
mean_reds = []
mean_greens = []
mean_blues = []
texture_stds = []
num_faces_list = []
aspect_ratios = []
laplacian_variances = []

# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    # Construct the full path to the image file
    image_path = os.path.join(folder_path, filename)
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image '{image_path}'. Skipping...")
        continue
    
    # Calculate mean values of red, green, and blue channels
    mean_red = int(image[:, :, 0].mean())
    mean_green = int(image[:, :, 1].mean())
    mean_blue = int(image[:, :, 2].mean())
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate standard deviation of pixel values in grayscale image
    texture_std = int(gray_image.std())
    
    # Perform face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    num_faces = len(faces)
    
    # Find the largest contour
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Calculate aspect ratio of bounding box
    aspect_ratio = int(w / h) if h != 0 else 0
    
    # Compute Laplacian variance
    laplacian_var = int(cv2.Laplacian(gray_image, cv2.CV_64F).var())
    
    # Append condition values to lists
    image_names.append(filename)
    mean_reds.append(mean_red)
    mean_greens.append(mean_green)
    mean_blues.append(mean_blue)
    texture_stds.append(texture_std)
    num_faces_list.append(num_faces)
    aspect_ratios.append(aspect_ratio)
    laplacian_variances.append(laplacian_var)

# Create DataFrame from lists
conditions_df = pd.DataFrame({
    'Image': image_names,
    'Mean_Red': mean_reds,
    'Mean_Green': mean_greens,
    'Mean_Blue': mean_blues,
    'Texture_Std': texture_stds,
    'Num_Faces': num_faces_list,
    'Aspect_Ratio': aspect_ratios,
    'Laplacian_Variance': laplacian_variances
})

# Save DataFrame to CSV
csv_filename = 'image_conditions.csv'
conditions_df.to_csv(csv_filename, index=False)

print(f"Conditions for each image saved to '{csv_filename}'.")
