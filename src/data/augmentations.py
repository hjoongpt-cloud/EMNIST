# file: src/utils/augment.py
import random
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import RandAugment, functional as TF
from torchvision.transforms import ToTensor, ToPILImage, Normalize
from PIL import Image
class CenterBoxTransform:
    """
    Detects the minimal bounding box around foreground pixels and centers it within the image.
    Uses Otsu's method by default to dynamically threshold the background.
    """
    def __init__(self, background_value: float = None, use_otsu: bool = True):
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()
        self.bg = background_value
        self.use_otsu = use_otsu

    def __call__(self, img: Image.Image) -> Image.Image:
        # Convert PIL to tensor [C,H,W]
        t = self.to_tensor(img)
        arr = t[0].numpy()  # assume single-channel

        # Determine mask of foreground
        if self.use_otsu:
            img_u8 = (arr * 255).astype(np.uint8)
            _, binary = cv2.threshold(img_u8, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = binary > 0
        else:
            bg_val = self.bg if self.bg is not None else 0.0
            mask = arr > (bg_val + 1e-6)

        if not mask.any():
            return img

        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        crop = arr[y0:y1+1, x0:x1+1]

        H, W = arr.shape
        box_h, box_w = crop.shape
        pad_h = H - box_h
        pad_w = W - box_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded = np.pad(crop,
                        ((top, bottom), (left, right)),
                        mode='constant',
                        constant_values=self.bg if self.bg is not None else 0)
        out_tensor = torch.from_numpy(padded).unsqueeze(0)
        return self.to_pil(out_tensor)

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

class AddNoise:
    def __init__(self, gauss_scale, salt_prob):
        self.gauss = gauss_scale; self.salt = salt_prob
    def __call__(self, t):
        noise = self.gauss * torch.randn_like(t)
        salt = (torch.rand_like(t) < self.salt).float()
        return t + noise + salt

class ElasticTransform:
    def __init__(self, alpha, sigma, prob):
        self.alpha = alpha; self.sigma = sigma; self.prob = prob
        self.to_pil = ToPILImage(); self.to_tensor = ToTensor()
    def __call__(self, img):
        is_tensor = isinstance(img, torch.Tensor)
        pil = self.to_pil(img) if is_tensor else img
        if random.random()>self.prob:
            out = pil
        else:
            arr = np.array(pil)
            dx = cv2.GaussianBlur((np.random.rand(*arr.shape)*2-1).astype(np.float32),(0,0),self.sigma)*self.alpha
            dy = cv2.GaussianBlur((np.random.rand(*arr.shape)*2-1).astype(np.float32),(0,0),self.sigma)*self.alpha
            x,y = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
            map_x = (x+dx).astype(np.float32); map_y = (y+dy).astype(np.float32)
            warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            out = Image.fromarray(warped)
        return self.to_tensor(out) if is_tensor else out

class RandomRotateTransform:
    def __init__(self, degrees, prob): self.degrees=degrees; self.prob=prob
    def __call__(self, img):
        if random.random()>self.prob: return img
        angle = random.uniform(-self.degrees, self.degrees)
        return TF.rotate(img, angle)

class RandomTranslateTransform:
    def __init__(self, max_dx, max_dy, prob): self.max_dx=max_dx; self.max_dy=max_dy; self.prob=prob
    def __call__(self, img):
        if random.random()>self.prob: return img
        if isinstance(img, torch.Tensor): 
            _,H,W=img.shape
            dx=int(random.uniform(-self.max_dx,self.max_dx)*W)
            dy=int(random.uniform(-self.max_dy,self.max_dy)*H)
            return TF.affine(img, angle=0, translate=(dx,dy), scale=1.0, shear=0)
        W,H = img.size; dx=int(random.uniform(-self.max_dx,self.max_dx)*W); dy=int(random.uniform(-self.max_dy,self.max_dy)*H)
        return TF.affine(img, angle=0, translate=(dx,dy), scale=1.0, shear=0)

class RandomAffineTransform:
    def __init__(self, degrees, translate, scale, shear, prob):
        self.degrees=degrees;self.translate=translate;self.scale=scale;self.shear=shear;self.prob=prob
        self.to_pil=ToPILImage();self.to_tensor=ToTensor()
    def __call__(self,img):
        is_tensor=isinstance(img,torch.Tensor)
        pil=self.to_pil(img) if is_tensor else img
        if random.random()>self.prob: out=pil
        else:
            W,H=pil.size;angle=random.uniform(-self.degrees,self.degrees)
            tx=int(random.uniform(-self.translate[0],self.translate[0])*W)
            ty=int(random.uniform(-self.translate[1],self.translate[1])*H)
            scale=random.uniform(self.scale[0],self.scale[1]);shear=self.shear
            out=TF.affine(pil, angle=angle, translate=(tx,ty), scale=scale, shear=shear)
        return self.to_tensor(out) if is_tensor else out

class CutMixTransform:
    def __init__(self, prob, beta): self.prob=prob;self.beta=beta
    def __call__(self,batch): 
        images,labels=batch
        if random.random()>self.prob: 
            return images,labels
        lam=np.random.beta(self.beta,self.beta);B,_,H,W=images.size();idx=torch.randperm(B)
        cx,cy=np.random.randint(W),np.random.randint(H)
        w=int(W*np.sqrt(1-lam));h=int(H*np.sqrt(1-lam))
        x1,x2=np.clip(cx-w//2,0,W),np.clip(cx+w//2,0,W)
        y1,y2=np.clip(cy-h//2,0,H),np.clip(cy+h//2,0,H)
        images[:,:,y1:y2,x1:x2]=images[idx,:,y1:y2,x1:x2]
        labels=lam*labels+(1-lam)*labels[idx]
        return images,labels

class MixUpTransform:
    def __init__(self, prob, alpha): self.prob=prob;self.alpha=alpha
    def __call__(self,batch): 
        images,labels=batch
        if random.random()>self.prob: 
            return images,labels
        lam=np.random.beta(self.alpha,self.alpha);B=images.size(0);idx=torch.randperm(B)
        return lam*images+(1-lam)*images[idx], lam*labels+(1-lam)*labels[idx]

# build and batch augment functions
AUG_IMAGE = {
    'rotate': RandomRotateTransform,
    'translate': RandomTranslateTransform,
    'affine': RandomAffineTransform,
    'add_noise': AddNoise,
    'elastic': ElasticTransform,
    'randaugment': lambda num_ops, magnitude: RandAugment(num_ops=num_ops, magnitude=magnitude)
}
AUG_BATCH = {
    'cutmix': CutMixTransform,
    'mixup': MixUpTransform
}

def build_augment_pipeline(augment_cfg: dict):
    """
    Build image-level augment pipeline based on augment_cfg.
    Separates PIL- and tensor-based transforms to ensure correct types.
    Backward-compatible with 'invert_if_smart' flag.
    """
    strength = augment_cfg.get('strength', 'light')
    presets  = augment_cfg.get('presets', {})
    params   = augment_cfg.get('params', {})
    names    = presets.get(strength, [])

    pil_augs   = []
    tensor_augs = []
    # backward-compatible invert_if_smart flag
    if augment_cfg.get('invert_if_smart', False):
        pil_augs.append(InvertIfSmart(**params.get('invert_if_smart', {})))
    # explicit invert_params override
    if 'invert_params' in augment_cfg:
        pil_augs.append(InvertIfSmart(**augment_cfg['invert_params']))

    # split transforms by type
    for name in names:
        if name in AUG_IMAGE:
            cls = AUG_IMAGE[name]
            kwargs = params.get(name, {})
            if name == 'add_noise':
                tensor_augs.append(cls(**kwargs))
            else:
                pil_augs.append(cls(**kwargs))

    # compose: PIL augs -> tensor conversion -> tensor augs -> normalize
    pipeline = pil_augs + [ToTensor()] + tensor_augs + [Normalize((0.5,), (0.5,))]
    # return a composed transform pipeline, not apply it to undefined list
    return transforms.Compose(pipeline)

def apply_batch_augment(images, labels, augment_cfg):
    strength = augment_cfg.get('strength', 'light')
    presets  = augment_cfg.get('presets', {})
    params   = augment_cfg.get('params', {})
    names    = presets.get(strength, [])

    imgs, lbls = images, labels

    # If using cutmix/mixup, convert labels to one-hot float vectors
    if any(name in AUG_BATCH for name in names):
        # get num_classes from your model config
        num_classes = augment_cfg.get('model', {}).get('num_classes')
        lbls = F.one_hot(lbls, num_classes).float()

    # apply each batch-level aug in sequence
    for name in names:
        if name in AUG_BATCH:
            cls = AUG_BATCH[name]
            imgs, lbls = cls(**params.get(name, {}))((imgs, lbls))

    return imgs, lbls