import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import matplotlib.image as mpimg
import os
from scipy.ndimage import gaussian_filter

# Load multiple images from a folder
image_folder = '/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/archive'  # Replace with your folder path

# Get all image file paths in the folder (supports PNG, JPG, JPEG)
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

img_stack = []
img_stack_blur = []

# Plot setup
fig, ax = plt.subplots(figsize=(12, 10))

# Variables to store transformed image bounds for setting the final plot limits
overall_x_min, overall_y_min = float('inf'), float('inf')
overall_x_max, overall_y_max = float('-inf'), float('-inf')

# Loop through the images and apply transformations
for idx, image_file in enumerate(image_files):
    # Load the image
    img = mpimg.imread(image_file)
    img_height, img_width = img.shape[:2]

    # gausian blur using matplotlib
    if idx in [0, 9]:
        img = gaussian_filter(img, sigma=10)
    if idx in [1, 8]:
        img = gaussian_filter(img, sigma=8)
    if idx in [2, 7]:
        img = gaussian_filter(img, sigma=3)
    if idx in [3, 6]:
        img = gaussian_filter(img, sigma=2)
    if idx in [4]:
        img = gaussian_filter(img, sigma=0.5)


    img_stack.append(img)
    # calculate the sharpness of the image
    sharpness = np.var(img)
    print(f"Sharpness of image {idx}: {sharpness}")
    img_stack_blur.append(sharpness)


    # Apply transformations with -30 degree rotation and vertical offset
    transform = (
        Affine2D()
        .rotate_deg(15)  # Rotate by -30 degrees
        .skew_deg(-50, 0)   # No skew applied
        .translate(0, idx * 150)  # Offset images vertically
    )

    # Calculate the transformed corners to determine plot limits
    corners = np.array([
        [0, 0],
        [img_width, 0],
        [img_width, img_height],
        [0, img_height]
    ])
    transformed_corners = transform.transform(corners)

    # Update the overall bounds for the plot
    x_min, y_min = transformed_corners.min(axis=0)
    x_max, y_max = transformed_corners.max(axis=0)

    overall_x_min = min(overall_x_min, x_min)
    overall_y_min = min(overall_y_min, y_min)
    overall_x_max = max(overall_x_max, x_max)
    overall_y_max = max(overall_y_max, y_max)

    # Display the image on the plot
    ax.imshow(img, cmap='gray', transform=transform + ax.transData, alpha=1, zorder=len(image_files) - idx)

# Adjust plot limits dynamically based on transformed image bounds
ax.set_xlim(overall_x_min, overall_x_max)
ax.set_ylim(overall_y_max, overall_y_min)  # Inverted y-axis for correct display
ax.axis('off')  # Turn off the axis for a cleaner look

# Save the output image with a transparent background
output_file = '/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/output_transparent.png'
fig.savefig(output_file, transparent=True, bbox_inches='tight', dpi=300)


# plot the sharpness of the images in a line plot
# interplot the sharpness curve to 10 points
# img_stack_blur = np.interp(np.linspace(0, len(img_stack_blur), 10), np.arange(len(img_stack_blur)), img_stack_blur)


fig2, ax2 = plt.subplots()
ax2.plot(img_stack_blur, color='w', marker='o', markersize=10) 
ax2.set_title('Sharpness of Images', color='w', fontsize=32)
ax2.set_xlabel('Image Index', color='w')
ax2.set_ylabel('Sharpness', color='w')
# remove axis and frame and ticks
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.yaxis.set_ticks_position('none')
ax2.xaxis.set_ticks_position('none')
ax2.yaxis.set_tick_params(width=0)
ax2.xaxis.set_tick_params(width=0)
ax2.yaxis.set_visible(False)
ax2.xaxis.set_visible(False)



# svae the output image with a transparent background
output_file2 = '/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/output_sharpness.png'
fig2.savefig(output_file2, transparent=True, bbox_inches='tight', dpi=300)


