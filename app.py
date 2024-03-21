import os
import random
from flask import Flask, render_template, request, send_file
from PIL import Image

app = Flask(__name__)

# Directory where the images are stored
image_directory = 'C:/Users/tariq/Desktop/Tariq/PNW/Spring 2024/ITS 365/KNN/archive/train/nascar racing'

# Route to display the form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and generate image
@app.route('/generate', methods=['POST'])
def generate_image():
    # Extract user-inputted conditions from form data
    mean_red = int(request.form['mean_red'])
    mean_green = int(request.form['mean_green'])
    mean_blue = int(request.form['mean_blue'])
    # Add more conditions as needed (not added yet)

    # Get a list of all image filenames in the directory
    image_files = os.listdir(image_directory)

    # Select two random images from the directory
    image1_filename, image2_filename = random.sample(image_files, 2)

    # Load the selected images
    image1 = Image.open(os.path.join(image_directory, image1_filename))
    image2 = Image.open(os.path.join(image_directory, image2_filename))

    # Perform any image processing based on user inputs
    # For demonstration, let's create a new blank image with the specified color
    generated_image = Image.new('RGB', (100, 100), color=(mean_red, mean_green, mean_blue))

    # Save the generated image to a temporary file
    temp_image_path = 'temp_generated_image.jpg'
    generated_image.save(temp_image_path)

    # Serve the generated image back to the user
    return send_file(temp_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
