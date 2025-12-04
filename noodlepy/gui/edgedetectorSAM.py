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

class EdgeDetectorSAM():
    def __init__(self, image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        self.width, self.height = image.shape[1], image.shape[0]
        self.marker_size = image.shape[1]/2
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        model_path = os.path.join(os.getcwd(), "sam_vit_h_4b8939.pth")
        self.sam = sam_model_registry["vit_h"](checkpoint=model_path)
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
        save_path = os.path.join(os.getcwd(), "masked_image.png")
        base_name, ext = os.path.splitext(save_path)
        index = 1
        while os.path.exists(save_path):
            save_path = f"{base_name}_{index}{ext}"
            index += 1
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

        elif shape == 'line':
            num_points = kwargs['num_points']

            # ensure mask has positive pixels
            if not mask.any():
                sampling_x, sampling_y = np.array([]), np.array([])
            else:
                h_img, w_img = mask.shape
                # centroid of mask pixels
                ys, xs = np.where(mask == 1)
                cx = int(np.round(xs.mean()))
                cy = int(np.round(ys.mean()))

                # search outward from centroid for a row with mask pixels
                found_row = None
                for offset in range(h_img):
                    for r in (cy + offset, cy - offset) if offset else (cy,):
                        if 0 <= r < h_img:
                            cols = np.where(mask[r] == 1)[0]
                            if cols.size:
                                # pick the largest contiguous segment in this row
                                diffs = np.diff(cols)
                                splits = np.where(diffs > 1)[0]
                                segments = []
                                start = 0
                                for s in splits:
                                    segments.append(cols[start:s+1])
                                    start = s + 1
                                segments.append(cols[start:])
                                largest = max(segments, key=len)
                                xmin, xmax = float(largest[0]), float(largest[-1])
                                sampling_x = np.linspace(xmin, xmax, num_points)
                                sampling_y = np.full_like(sampling_x, float(r))
                                found_row = r
                                break
                    if found_row is not None:
                        break

                # fallback if no row found (use full mask horizontal span)
                if found_row is None:
                    xmin, xmax = float(xs.min()), float(xs.max())
                    sampling_x = np.linspace(xmin, xmax, num_points)
                    sampling_y = np.full_like(sampling_x, float(self.center_y))

        elif shape == 'rings':
            num_points = kwargs['num_points']
            num_rings = kwargs['num_rings']
            interval = kwargs['interval']
            offset_from_the_edge = kwargs['offset_from_the_edge']
            sampling_x, sampling_y = np.array([]), np.array([])
            
            for i_ring in range(num_rings):
                erosion_size = interval * i_ring + offset_from_the_edge
                edge_coords = self.find_edge_of_eroded_mask(mask, erosion_size=erosion_size, erosion_shape=cv.MORPH_RECT)
                
                if len(edge_coords) == 0:
                    continue
                
                # Sample points with consistent spacing along the contour
                sampled_points = self.sample_points_along_contour(edge_coords, num_points)
                
                if len(sampled_points) > 0:
                    y_i_ring, x_i_ring = sampled_points[:, 0], sampled_points[:, 1]
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
    def sample_points_along_contour(edge_coords, num_points):
        """
        Sample points along a contour with consistent spacing.
        
        Parameters:
            edge_coords (np.ndarray): Coordinates of the edge points.
            num_points (int): Number of points to sample.
            
        Returns:
            sampled_points (np.ndarray): Sampled points with consistent spacing.
        """
        if len(edge_coords) < 2:
            return edge_coords
        
        # Find contours and get the longest one
        edge_image = np.zeros((edge_coords[:, 0].max() + 1, edge_coords[:, 1].max() + 1), dtype=np.uint8)
        edge_image[edge_coords[:, 0], edge_coords[:, 1]] = 255
        
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return edge_coords[:num_points] if len(edge_coords) > num_points else edge_coords
        
        # Get the longest contour
        longest_contour = max(contours, key=cv2.contourArea)
        
        # Reshape contour to (N, 2) format
        contour_points = longest_contour.reshape(-1, 2)
        
        # Calculate cumulative distances along the contour
        distances = np.zeros(len(contour_points))
        for i in range(1, len(contour_points)):
            dist = np.linalg.norm(contour_points[i] - contour_points[i-1])
            distances[i] = distances[i-1] + dist
        
        # Total perimeter
        total_distance = distances[-1]
        
        # Calculate target distances for evenly spaced points
        target_distances = np.linspace(0, total_distance, num_points, endpoint=False)
        
        # Find points at target distances
        sampled_points = []
        for target_dist in target_distances:
            # Find the segment containing this distance
            idx = np.searchsorted(distances, target_dist)
            
            if idx == 0:
                point = contour_points[0]
            elif idx >= len(contour_points):
                point = contour_points[-1]
            else:
                # Interpolate between points
                segment_start_dist = distances[idx-1]
                segment_end_dist = distances[idx]
                t = (target_dist - segment_start_dist) / (segment_end_dist - segment_start_dist)
                
                # Linear interpolation
                point = (1 - t) * contour_points[idx-1] + t * contour_points[idx]
            
            # Convert from (x, y) to (y, x) format to match edge_coords
            sampled_points.append([point[1], point[0]])
        
        return np.array(sampled_points)

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

    edge_detector = EdgeDetectorSAM(image)
    pil_image = edge_detector.point_prompt_mask_generate()
    pil_image.show()
    save_path = "masked_image.png"
    base_name, ext = os.path.splitext(save_path)
    index = 1
    while os.path.exists(save_path):
        save_path = f"{base_name}_{index}{ext}"
        index += 1
    pil_image.save(save_path)
    print(f"Mask generated successfully! Saved as {save_path}")

    image, x, y = edge_detector.generate_sampling_points(shape='rings', num_points=50, num_rings=3, interval=10, offset_from_the_edge=5)