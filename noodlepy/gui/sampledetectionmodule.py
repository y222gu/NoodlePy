import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
import ttkbootstrap as ttk


class SampleDetectionModule(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_widgets()

    def create_widgets(self):
        self.image_path = ttk.Entry(self)
        self.image_path.pack()

        self.auto_mask_button = ttk.Button(self, text="Auto Mask Generate", command=self.auto_mask_generate)
        self.auto_mask_button.pack()

        self.point_prompt_button = ttk.Button(self, text="Point Prompt Mask Generate", command=self.point_prompt_mask_generate)
        self.point_prompt_button.pack()

        self.box_prompt_button = ttk.Button(self, text="Box Prompt Mask Generate", command=self.box_prompt_mask_generate)
        self.box_prompt_button.pack()

        self.canvas = ttk.Canvas(self)
        self.canvas.pack()


class SampleDetector(SamPredictor):
    def __init__(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        sam = sam_model_registry["vit_b"](checkpoint="/Users/yifeigu/Downloads/sam_vit_b_01ec64.pth")
        self.auto_mask_generator = SamAutomaticMaskGenerator(model=sam)
        self.prompted_mask_generator = SamPredictor(sam)

    def auto_mask_generate(self):
        masks = self.auto_mask_generator.generate(self.image)
        _, axes = plt.subplots(1,3, figsize=(16,16))
        axes[0].imshow(self.image)
        SampleDetector.show_anns(masks, axes[1])
        axes[2].imshow(self.image)
        SampleDetector.show_anns(masks, axes[2])
        plt.show()

    def point_prompt_mask_generate(self):
        input_point = np.array([[400, 400], [1200, 400], [400, 700]])
        input_label = np.array([1, 1, 0])
        self.prompted_mask_generator.set_image(self.image)
        masks, scores, logits = self.prompted_mask_generator.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.imshow(self.image)
            SampleDetector.show_mask(mask, plt.gca())
            SampleDetector.show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.show()  
  
    def box_prompt_mask_generate(self, x_low=950, y_low=180, x_high=1400, y_high=620):
        input_box = np.array([x_low, y_low, x_high, y_high])
        self.prompted_mask_generator.set_image(self.image)
        masks, scores, logits = self.prompted_mask_generator.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.imshow(self.image)
            SampleDetector.show_mask(mask, plt.gca())
            SampleDetector.show_box(input_box, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.show()

    def show_anns(anns, axes=None):
        if len(anns) == 0:
            return
        if axes:
            ax = axes
        else:
            ax = plt.gca()
            ax.set_autoscale_on(False)
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack((img, m*0.5)))

    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


if __name__ == '__main__':
        
    image_path = '/Users/yifeigu/Documents/Carney_Lab/DiddyKong/outputs/img1.png'
    sd = SampleDetector(image_path)
    sd.auto_mask_generate()
    sd.point_prompt_mask_generate()
    sd.box_prompt_mask_generate()


