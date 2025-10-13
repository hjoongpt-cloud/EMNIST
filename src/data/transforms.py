import random
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import RandAugment, functional as TF
from torchvision.transforms import ToTensor, ToPILImage, Normalize
from PIL import Image

# --- Augment Classes ---
class SobelGrayTransform:
    def __init__(self):
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        self.fx = kx.view(1,1,3,3)
        self.fy = ky.view(1,1,3,3)
    def __call__(self, img):
        t = transforms.ToTensor()(img).unsqueeze(0)
        gx = F.conv2d(t, self.fx, padding=1)
        gy = F.conv2d(t, self.fy, padding=1)
        mag = (gx**2 + gy**2).sqrt().squeeze(0)
        minv = mag.view(1,-1).min(1)[0].view(1,1,1)
        maxv = mag.view(1,-1).max(1)[0].view(1,1,1)
        return (mag - minv) / (maxv - minv + 1e-6)
    
class InvertIfSmart:
    def __init__(self, border_width=8, otsu_threshold=True, sobel_transform=None, sobel_thresh=0.3):
        self.border_width = border_width
        self.otsu_enabled = otsu_threshold
        self.sobel = sobel_transform or SobelGrayTransform()
        self.sobel_thresh = sobel_thresh
        self.to_tensor = transforms.ToTensor()
        self.to_pil = ToPILImage()

    def __call__(self, img):
        # Accept PIL Image, perform decision on tensor, then return PIL
        t = self.to_tensor(img)  # [C,H,W] float in [0,1]
        # border vs center mean
        _, H, W = t.shape
        bw = min(self.border_width, H//2, W//2)
        b = torch.cat([
            t[:, :bw, :].reshape(-1),
            t[:, H-bw:, :].reshape(-1),
            t[:, :, :bw].reshape(-1),
            t[:, :, W-bw:].reshape(-1)
        ])
        c = t[:, bw:H-bw, bw:W-bw].reshape(-1)

        # decide inversion
        invert = False
        if b.mean() > c.mean():
            invert = True
        elif self.otsu_enabled:
            img_np = (t.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            _, bin_img = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if (bin_img == 255).sum() > (bin_img == 0).sum():
                invert = True
        elif float(self.sobel(t).mean()) < self.sobel_thresh:
            invert = True

        t_final = 1.0 - t if invert else t
        return self.to_pil(t_final)