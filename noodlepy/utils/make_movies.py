from PIL import Image
import os
from IPython.display import display

# Path to the folder containing images
images_folder = "/Users/yifeigu/Documents/Carney_Lab/NoodlePy/scratch/"

# Get a list of image file names in the folder and sort them
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Create a list to store image frames
frames = []

# Iterate through the sorted image files and append them to the frames list
for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)
    frame = Image.open(image_path)
    
    # Resize the frame if needed
    # frame = frame.resize((640, 480))
    
    frames.append(frame)

# Specify the output GIF file path
output_path = "/Users/yifeigu/Documents/Carney_Lab/NoodlePy/scratch/animation.gif"

# Save the frames as a GIF animation
frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=500, loop=0)
