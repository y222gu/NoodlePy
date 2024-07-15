import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt

# Load the Segment Anything model
sam = sam_model_registry["vit_b"](checkpoint="/Users/yifeigu/Downloads/sam_vit_b_01ec64.pth")  # Replace with appropriate model name and path
mask_generator = SamAutomaticMaskGenerator(sam)

def segment_circles(image_path, min_radius, max_radius):
    # Load the image
    image = cv2.imread(image_path)

    # display the image
    cv2.imshow('img', image)
    cv2.waitKey(10)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img', gray)
    cv2.waitKey(10)
    
    # Use HoughCircles to detect circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        masks = []
        boundaries = []
        
        for (x, y, r) in circles:
            # Create a mask for the circle
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            mask = mask.astype(np.uint8)  # Ensure mask is of correct type
            
            # Use the mask with Segment Anything (Assuming it uses image and mask)
            masks_output = mask_generator.generate(image, mask)  # This line may need adjustment based on the correct API usage

            # Use the mask directly for now
            masks.append(mask)
            
            # Extract the boundary coordinates
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                boundary = contour[:, 0, :]
                boundaries.append(boundary.tolist())
                
        return image, masks, boundaries
    else:
        return None, None, None

def visualize_results(image, boundaries):
    # Draw boundaries on the image
    for boundary in boundaries:
        cv2.polylines(image, [np.array(boundary)], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Convert BGR to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

# Example usage
image_path = '/Users/yifeigu/Documents/Carney_Lab/DiddyKong/outputs/img1.png'
min_radius = 180  # minimum radius of circles to detect
max_radius = 250  # maximum radius of circles to detect

image, masks, boundaries = segment_circles(image_path, min_radius, max_radius)

if image is not None:
    print("Masks and boundaries detected:")
    for i, boundary in enumerate(boundaries):
        print(f"Boundary {i}: {boundary}")
    
    visualize_results(image, boundaries)
else:
    print("No circles detected.")
