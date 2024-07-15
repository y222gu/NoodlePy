import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


# Load the SAM model
sam = sam_model_registry["vit_b"](checkpoint="/Users/yifeigu/Downloads/sam_vit_b_01ec64.pth")
mask_generator = SamAutomaticMaskGenerator(
    model=sam  # Requires open-cv to run post-processing
)


# Example usage
image_path = '/Users/yifeigu/Documents/Carney_Lab/DiddyKong/outputs/img1.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 
