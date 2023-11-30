import os
import sys
sys.path.append(os.path.join(".."))

from PIL import Image

my_folder = os.path.join('/g', 'prevedel', 'members', 'Rauscher')
data_folder = os.path.join(my_folder, 'data', 'BSD300', 'clean', 'test')

# Get a list of all files in the folder
files = os.listdir(data_folder)

# Filter out only image files (you may need to adjust this based on your file types)
image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# Check if there are any image files in the folder
if image_files:
    # Get the path of the first image file
    first_image_path = os.path.join(data_folder, image_files[0])

    # Open the image using Pillow and get its type
    with Image.open(first_image_path) as img:
        img_type = type(img)

    # Print the variable type
    print(f"The variable type of the first image is: {img_type}")
else:
    print("No image files found in the folder.")


print(my_folder)