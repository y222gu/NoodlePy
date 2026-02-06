import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from PIL import Image
import segmentation_models_pytorch as smp

# ImageNet normalization constants (must match training)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default path to best model
DEFAULT_CHECKPOINT = "/Users/yifeigu/Documents/Carney_Lab/droplet_segmentation/best_model.pth"


class EdgeDetectorUnet:
    def __init__(self, image, checkpoint_path=None, encoder_name="resnet34", target_size=512):
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Handle both RGB and BGR input
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume input is BGR from cv2, convert to RGB
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            self.image = image

        self.height, self.width = self.image.shape[:2]
        self.best_mask = None
        self.masked_image = None
        self.target_size = target_size

        # Normalization constants
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(IMAGENET_STD, dtype=np.float32)

        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load U-Net model
        checkpoint_path = os.path.join(os.getcwd(), "utils", "unet_trained_20260205.pth")
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, image):
        """Preprocess image for model input."""
        img = cv2.resize(image, (self.target_size, self.target_size))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _postprocess_mask(self, mask):
        """Apply post-processing: largest component, hole fill, morphology."""
        # Keep largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8) * 255

        # Fill holes
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(mask, contours, -1, 255, -1)

        # Morphological close then open
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def point_prompt_mask_generate(self, threshold=0.5, postprocess=True, use_tta=False):
        """
        Generate segmentation mask using trained U-Net model.

        Args:
            threshold: Binarization threshold for mask
            postprocess: Whether to apply post-processing (largest component, hole fill, morphology)
            use_tta: Whether to use test-time augmentation (averages 4 flip variants)

        Returns:
            PIL Image with mask overlay
        """
        if use_tta:
            mask, confidence = self._predict_tta(threshold, postprocess)
        else:
            mask, confidence = self._predict(threshold, postprocess)

        if mask is None or not mask.any():
            print("No mask detected.")
            self.masked_image = None
            return None

        # Convert to boolean mask
        self.best_mask = mask > 0
        self.masked_image = self.plot_mask()
        return self.masked_image

    def _predict(self, threshold=0.5, postprocess=True):
        """Predict segmentation mask for the image."""
        original_size = (self.height, self.width)

        input_tensor = self._preprocess(self.image)

        with torch.no_grad():
            logits = self.model(input_tensor)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Resize probability map to original size
        prob = cv2.resize(prob, (original_size[1], original_size[0]))

        # Binarize
        mask = (prob > threshold).astype(np.uint8) * 255

        # Confidence
        confidence = float(prob[mask > 0].mean()) if mask.any() else 0.0

        if postprocess:
            mask = self._postprocess_mask(mask)

        return mask, confidence

    def _predict_tta(self, threshold=0.5, postprocess=True):
        """Predict with test-time augmentation (4 flip variants averaged)."""
        original_size = (self.height, self.width)

        # Generate 4 augmented versions
        augmented = [
            self.image,
            np.fliplr(self.image).copy(),
            np.flipud(self.image).copy(),
            np.flipud(np.fliplr(self.image)).copy(),
        ]

        prob_sum = np.zeros((self.target_size, self.target_size), dtype=np.float32)

        with torch.no_grad():
            for i, aug_img in enumerate(augmented):
                input_tensor = self._preprocess(aug_img)
                logits = self.model(input_tensor)
                prob = torch.sigmoid(logits).squeeze().cpu().numpy()

                # Undo the augmentation on the probability map
                if i == 1:
                    prob = np.fliplr(prob)
                elif i == 2:
                    prob = np.flipud(prob)
                elif i == 3:
                    prob = np.flipud(np.fliplr(prob))

                prob_sum += prob

        # Average predictions
        prob_avg = prob_sum / len(augmented)

        # Resize to original size
        prob_avg = cv2.resize(prob_avg, (original_size[1], original_size[0]))

        # Binarize
        mask = (prob_avg > threshold).astype(np.uint8) * 255

        confidence = float(prob_avg[mask > 0].mean()) if mask.any() else 0.0

        if postprocess:
            mask = self._postprocess_mask(mask)

        return mask, confidence
        
    def plot_mask(self):
        if self.best_mask is None:
            print("No mask available to plot.")
            return None
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
        self.show_mask(self.best_mask, plt.gca())
        plt.axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        pil_image = Image.open(buf)
        pil_image.save(os.path.join(os.getcwd(), "masked_image.png"))
        plt.close()
        return pil_image

    @staticmethod
    def show_mask(mask, ax, random_color=False):
        color = np.concatenate([np.random.rand(3), [0.6]]) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
        mask_image = np.zeros((*mask.shape, 4))
        mask_image[mask] = color
        ax.imshow(mask_image)
