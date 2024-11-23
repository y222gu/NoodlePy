import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
import os
from PIL import Image
import io
import torch
import cv2 as cv
from threading import Thread

class EdgeDetector():
    def __init__(self, image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        self.width, self.height = image.shape[1], image.shape[0]
        self.marker_size = image.shape[1]/2
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        model_path = os.path.join(os.getcwd(), "sam_vit_b_01ec64.pth")
        self.sam = sam_model_registry["vit_b"](checkpoint=model_path)
        if torch.cuda.is_available():
            self.sam.to('cuda')

    def auto_mask_generate(self):
        generator = SamAutomaticMaskGenerator(model=self.sam)
        masks = generator.generate(self.image)

        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(self.width*px, self.height*px))
        ax.imshow(self.image)
        self.show_anns(masks, ax)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')  # Save the figure to the buffer
        buf.seek(0)  # Rewind the buffer
        pil_image = Image.open(buf)

        # save the figure
        # plt.savefig(os.path.join(os.getcwd(), "masked_image.png"))
        plt.close(fig)
        return pil_image


    def point_prompt_mask_generate(self, label=1):
        # put the point in the center of the image
        prompt_x, prompt_y = self.center_x, self.center_y

        input_point = np.array([[prompt_x, prompt_y]])
        input_label = np.array([label])
        generator = SamPredictor(self.sam)
        generator.set_image(self.image)
        masks, scores, logits = generator.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        # second mask is the usually best mask
        self.best_mask = masks[1]
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(self.width*px, self.height*px))
        ax.imshow(self.image)
        self.show_mask(self.best_mask, ax)
        self.show_points(input_point, input_label, ax, '*', marker_size=self.marker_size)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')  # Save the figure to the buffer
        buf.seek(0)  # Rewind the buffer
        pil_image = Image.open(buf)

        # save the figure
        save_path = os.path.join(os.getcwd(), f"masked_image.png")
        pil_image.save(save_path)
        plt.close(fig)
        return pil_image

    def box_prompt_mask_generate(self, x_low=950, y_low=180, x_high=1400, y_high=620):
        input_box = np.array([x_low, y_low, x_high, y_high])
        generator = SamPredictor(self.sam)
        generator.set_image(self.image)
        masks, scores, logits = generator.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(self.width*px, self.height*px))
        ax.imshow(self.image)
        self.show_mask(masks[0], ax)
        self.show_box(input_box, ax)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')  # Save the figure to the buffer
        buf.seek(0)  # Rewind the buffer
        pil_image = Image.open(buf)

        # save the figure
        # plt.savefig(os.path.join(os.getcwd(), "masked_image.png"))
        plt.close(fig)
        return pil_image
        

    def generate_sampling_points(self, shape, **kwargs):

        h, w = self.best_mask.shape[-2:]
        mask = self.best_mask.reshape(h, w)
        mask = mask > 0.5
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        sampling_y, sampling_x = np.where(mask == 1)

        if shape == 'grid':
            row_number, col_number = kwargs['row_number'], kwargs['col_number']
            sampling_x = np.linspace(sampling_x.min(), sampling_x.max(), int(col_number))
            sampling_y = np.linspace(sampling_y.min(), sampling_y.max(), int(row_number))
            sampling_x, sampling_y = np.meshgrid(sampling_x, sampling_y)
            sampling_x, sampling_y = sampling_x.flatten(), sampling_y.flatten()

        elif shape == 'random':
            num_points = kwargs['num_points']
            indices = np.random.choice(len(sampling_x), num_points, replace=False)
            sampling_x, sampling_y = sampling_x[indices], sampling_y[indices]

        elif shape == 'edge':
            num_points = kwargs['num_points']
            mask_edge = cv2.Canny(mask, 0, 1)
            edge_coords = np.argwhere(mask_edge > 0)
            indices = np.linspace(0, len(edge_coords) - 1, num_points).astype(int)
            sampling_y, sampling_x = edge_coords[indices].T

        elif shape == 'rings':
            num_points = kwargs['num_points']
            num_rings = kwargs['num_rings']
            interval = kwargs['interval']
            sampling_x, sampling_y = np.array([]), np.array([])
            for i_ring in range(num_rings):
                erosion_size = interval*i_ring
                edge_coords = self.find_edge_of_eroded_mask(mask, erosion_size=erosion_size, erosion_shape=cv.MORPH_RECT)
                indices = np.linspace(0, len(edge_coords) - 1, num_points).astype(int)
                y_i_ring, x_i_ring = edge_coords[indices].T
                sampling_x = np.concatenate([sampling_x, x_i_ring])
                sampling_y = np.concatenate([sampling_y, y_i_ring])

        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(self.width*px, self.height*px))
        ax.imshow(self.image)
        self.show_mask(self.best_mask, ax)
        self.show_points(np.stack([sampling_x, sampling_y], axis=1), np.ones(len(sampling_x)), ax, '.', marker_size=self.marker_size)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        pil_image = Image.open(buf)
        plt.close(fig)
        return pil_image, sampling_x, sampling_y

    @staticmethod
    def find_edge_of_eroded_mask(mask, erosion_size=10, erosion_shape=cv.MORPH_RECT):
        """
        Finds the edge coordinates of an eroded mask.
        
        Parameters:
            mask (np.ndarray): A binary matrix (2D numpy array) where the area of interest is labeled with 1.
            erosion_size (int): Size of the erosion kernel. Default is 3.
            erosion_shape (int): Shape of the erosion kernel. Options are cv.MORPH_RECT, cv.MORPH_CROSS, and cv.MORPH_ELLIPSE.
                                Default is cv.MORPH_RECT.
        
        Returns:
            edge_coords (np.ndarray): Coordinates of the edges in the eroded mask.
        """
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)
        element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size))
        eroded_mask = cv.erode(mask, element)
        edges = cv.Canny(eroded_mask, 0, 1)
        edge_coords = np.argwhere(edges > 0)
        return edge_coords


    @staticmethod
    def show_anns(anns, ax=None):
        if len(anns) == 0:
            return
        if ax is None:
            ax = plt.gca()
            ax.set_autoscale_on(False)
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m * 0.5)))

    @staticmethod
    def show_points(coords, labels, ax, marker, marker_size):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='#f39c12', marker=marker, s=marker_size, linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker=marker, s=marker_size, linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    @staticmethod
    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

if __name__ == "__main__":
    image_path = os.path.join(os.getcwd(), "livecamera_captured_frame.png")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    edge_detector = EdgeDetector(image)
    pil_image = edge_detector.point_prompt_mask_generate()
    pil_image.show()
    pil_image.save("masked_image.png")
    print("Mask generated successfully!")

    image, x, y = edge_detector.generate_sampling_points(shape='grid', row_number=4, col_number=4)