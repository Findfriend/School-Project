import os
import cv2
import pandas as pd
import numpy as np

# Path to the CSV file containing image conditions
conditions_file = 'C:/Users/Tariq/Desktop/Tariq/PNW/Spring 2024/ITS 365/KNN/image_conditions.csv'

# Check if the conditions file exists
if not os.path.exists(conditions_file):
    print(f"Error: Conditions file '{conditions_file}' not found.")
    exit()

# Load conditions from the CSV file
conditions_df = pd.read_csv(conditions_file)

# Path to the folder containing car images
folder_path = r'C:\Users\Tariq\Desktop\Tariq\PNW\Spring 2024\ITS 365\KNN\archive\train\nascar racing'

# Check if the folder exists
if not os.path.exists(folder_path):
    print(f"Error: Folder '{folder_path}' does not exist.")
    exit()

# Number of images to generate
num_images = 10

# Output folder to save generated images
output_folder = 'generated_images'
os.makedirs(output_folder, exist_ok=True)

# Iterate to generate new images
for i in range(num_images):
    # Randomly select a row (condition) from the conditions dataframe
    random_condition = conditions_df.sample(n=1)

    # Extract conditions from the selected row
    mean_red = random_condition['Mean_Red'].values[0]
    mean_green = random_condition['Mean_Green'].values[0]
    mean_blue = random_condition['Mean_Blue'].values[0]
    texture_std = random_condition['Texture_Std'].values[0]
    num_faces = random_condition['Num_Faces'].values[0]
    aspect_ratio = random_condition['Aspect_Ratio'].values[0]
    laplacian_variance = random_condition['Laplacian_Variance'].values[0]

    # Randomly select two car images as the base
    car_image_filenames = np.random.choice(os.listdir(folder_path), size=2, replace=False)
    car_images = [cv2.imread(os.path.join(folder_path, filename)) for filename in car_image_filenames]
    
    # Divide each image into two halves
    half_height = car_images[0].shape[0] // 2
    half_width = car_images[0].shape[1]
    
    upper_half_image1 = car_images[0][:half_height, :]
    lower_half_image2 = car_images[1][half_height:, :]
    
    # Combine the selected halves to create a new image
    combined_image = np.vstack((upper_half_image1, lower_half_image2))
    
    # Apply conditions to the combined image
    # Example: Modify mean RGB values, texture standard deviation, etc.
    combined_image[:, :, 0] = np.clip(combined_image[:, :, 0] + mean_red, 0, 255)
    combined_image[:, :, 1] = np.clip(combined_image[:, :, 1] + mean_green, 0, 255)
    combined_image[:, :, 2] = np.clip(combined_image[:, :, 2] + mean_blue, 0, 255)
    # Other conditions can be applied similarly

    # Save the combined image with conditions applied
    generated_image_filename = f'generated_image_{i+1}.jpg'
    cv2.imwrite(os.path.join(output_folder, generated_image_filename), combined_image)

print(f"{num_images} images generated and saved to '{output_folder}'.")
