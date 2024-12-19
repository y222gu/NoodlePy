import cv2
import os
import matplotlib.pyplot as plt
from skimage import io, filters

def is_image_blurry_opencv(images_folder):
    images_folder = "noodlepy/archive"

    files = os.listdir(images_folder)

    # list all png files in the folder
    image_files = [f for f in files if f.endswith(".png")]

    def sort_key(filename):
        return int(filename.split(".")[0].split("_")[-1])

    image_files.sort(key=sort_key)

    var_arr = []
    sharpness_arr = []

    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return "Invalid image path."
        print(image_file)
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        var_arr.append(laplacian_var)

        sobel_edges = filters.sobel(image)
        sharpness = sobel_edges.var()
        sharpness_arr.append(sharpness)

    # normalize the variance and sharpness values
    max_var = max(var_arr)
    max_var_index = var_arr.index(max_var)
    var_arr = var_arr/max_var

    max_sharpness = max(sharpness_arr)
    max_sharpness_index = sharpness_arr.index(max_sharpness)
    sharpness_arr = sharpness_arr/max_sharpness

    plt.plot(var_arr, 'b', label="Laplacian Variance")
    plt.plot(sharpness_arr, 'g', label="Sharpness")
    plt.title("Laplacian Variance of Images")
    plt.xlabel("Images")
    plt.ylabel("Laplacian Variance")
    plt.plot(max_var_index, 1, 'ro')
    plt.plot(max_sharpness_index, 1, 'ro')
    plt.legend() 
    plt.show()

    return max_var_index, max_sharpness_index


if __name__ == "__main__":
    folder = "noodlepy/archive"
    print(is_image_blurry_opencv(folder))

