from PIL import Image
import os

def check_images(directory):
    """
    Check all images in a directory for corruption.
    :param directory: Path to the directory containing images.
    """
    for subdir, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            
            # Check if the file is an image (based on extension for simplicity)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                try:
                    with Image.open(file_path) as img:
                        # Attempt to open and load the image
                        img.verify()
                except (IOError, SyntaxError) as e:
                    print(f"Corrupted image {file_path}: {e}")

# Use the function

img_path = f'/Users/dovydasluksa/Documents/Project_MSc/Train_img_final/y-axis_FL15_r60/renders'

check_images(img_path)
