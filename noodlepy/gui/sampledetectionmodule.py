import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
import ttkbootstrap as ttk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class SampleDetectionModule(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        image_path = r'/Users/yifeigu/Documents/Carney_Lab/NoodlePy/captured_frame_1.png'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        self.sam = sam_model_registry["vit_b"](checkpoint=r"/Users/yifeigu/Documents/Carney_Lab/NoodlePy/noodlepy/utils/sam_vit_b_01ec64.pth")
        # self.sam.to('cuda')
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Labelframe(self, text='Sample Detection', padding=5)
        main_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.imshow(self.image)
        ax.axis('off')  # Turn off the axis

        self.canvas = FigureCanvasTkAgg(figure, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, rowspan=2, sticky="nsew", padx=5, pady=5)

        self.auto_mask_button = ttk.Button(self, text="Auto Mask Generate", command=self.auto_mask_generate)
        self.auto_mask_button.grid(row=1, column=0, padx=5, pady=5)

        self.point_prompt_button = ttk.Button(self, text="Point Prompt Mask Generate", command=self.point_prompt_mask_generate)
        self.point_prompt_button.grid(row=1, column=1, padx=5, pady=5)

        self.box_prompt_button = ttk.Button(self, text="Box Prompt Mask Generate", command=self.box_prompt_mask_generate)
        self.box_prompt_button.grid(row=1, column=2, padx=5, pady=5)

    def auto_mask_generate(self):
        generator = SamAutomaticMaskGenerator(model=self.sam)
        masks = generator.generate(self.image)
        _, axes = plt.subplots(1,3, figsize=(16,16))
        axes[0].imshow(self.image)
        SampleDetectionModule.show_anns(masks, axes[1])
        axes[2].imshow(self.image)
        SampleDetectionModule.show_anns(masks, axes[2])
        plt.show()

    def point_prompt_mask_generate(self):
        input_point = np.array([[650, 600]])
        input_label = np.array([1])
        generator = SamPredictor(self.sam)
        generator.set_image(self.image)
        masks, scores, logits = generator.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        for i, (mask, score) in enumerate(zip(masks, scores)):
            # reverse mask
            # mask = 1 - mask
            plt.imshow(self.image)
            SampleDetectionModule.show_mask(mask, plt.gca())
            SampleDetectionModule.show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.show()  
  
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
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.imshow(self.image)
            SampleDetectionModule.show_mask(mask, plt.gca())
            SampleDetectionModule.show_box(input_box, plt.gca())
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

    root = ttk.Window()
    root.style.theme_use('superhero')
    app = SampleDetectionModule(root)
    app.pack()
    root.mainloop()



