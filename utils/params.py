import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from utils.ops import Jitter, ColorJitter, get_transforms, lucid_colorspace_to_rgb, bchw_rgb_to_fft_param, GaussianNoise


class ImageParams(torch.nn.Module):
    def __init__(self, init_image=None, image_size=128, std=0.01, rotate_degrees=15, scale_max=1.2, scale_min=0.5, \
                translate_x = 0.15, translate_y = 0.15, batch_size=1, max_iter=512, device="cuda", color_aug=False):
        super().__init__()
        self.image_size = image_size
        self.color_aug = color_aug
        self.max_iter = max_iter
        self.device = device
        self.optimizer = None
        self.lr_scheduler = None
        
        if init_image is None:
            img_buf = np.random.normal(size=(batch_size, 3, image_size, image_size), scale=std).astype(np.float32)
            # img_buf = np.zeros((batch_size, 3, image_size, image_size)).astype(np.float32)
            self.spectrum_t = torch.tensor(img_buf).float().to(device)
            self.lucid_space = True
        else:
            assert init_image.shape==(batch_size, 3, image_size, image_size)
            self.spectrum_t = bchw_rgb_to_fft_param(init_image).float().to(device)
            scale = self.get_fft_scale(round=True)
            self.spectrum_t = self.spectrum_t / scale
            self.lucid_space = False

        self.transform = get_transforms(image_size, rotate_degrees, scale_max, scale_min, translate_x, translate_y)
        if color_aug:
            self.jitter = Jitter(32)
            self.colorshift = ColorJitter(batch_size, True, mean=1.0, std=1.0, device=device)
        self.gaussian_noise = GaussianNoise(batch_size, True, device=device, std=0.5 if color_aug else 0.2, max_iter=max_iter)
        self.spectrum_t.requires_grad_(True)

    def set_transform(self, rotate_degrees, scale_max, scale_min, translate_x, translate_y):
        self.transform = get_transforms(self.image_size, rotate_degrees, scale_max, scale_min, translate_x, translate_y)

    def scaling(self, image):
        return F.interpolate(image, (self.image_size, self.image_size), mode="bilinear", align_corners=True)

    def forward(self, training=True, decay_power=0.75):
        return self.postprocess(training, decay_power=decay_power)
    
    def clip_grads(self, grad_clip=1.0):
        torch.nn.utils.clip_grad_norm_(self.spectrum_t, grad_clip)

    def postprocess(self, training=True, decay_power=0.75,):
        if self.lucid_space:
            img = self.fft_to_rgb(decay_power)
        else:
            scale = self.get_fft_scale(decay_power, True)
            img = scale * self.spectrum_t
            img = torch.fft.irfft2(img, s=(self.image_size, self.image_size), norm="ortho")
            
        img = lucid_colorspace_to_rgb(img, self.device)
        img = torch.sigmoid(img) if self.lucid_space else img.clamp(0,1)
            
        # self.transform(
        if training:
            img = self.transform(img)
            if self.color_aug:
                img = self.colorshift(self.jitter(img))
            img = self.gaussian_noise(img)
        return img

    def get_param_list(self):
        return [self.spectrum_t]

    def get_warmup_lr_scheduler(self, init_lr, lr_rampdown_length=0.25, lr_rampup_length=0.05):
        def lr_scheduler(step, num_steps):
            t = step / num_steps
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = init_lr * lr_ramp
            self.set_lr(lr)

        return lr_scheduler

    def get_optimizer_and_scheduler(self, param_list, lr, weight_decay=1e-2, warmup=False, eta_min=2e-4):
        self.optimizer = torch.optim.AdamW(param_list, lr=lr, betas=(0.5, 0.99), eps=1e-8, weight_decay=weight_decay)
        if warmup:
            self.lr_scheduler = self.get_warmup_lr_scheduler(lr)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_iter, eta_min=eta_min)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def to_chw_tensor(self):
        return self.postprocess(training=False).detach()

    def get_fft_scale(self, decay_power=0.75, round=False):
        d=.5**.5 # set center frequency scale to 1
        fy = np.fft.fftfreq(self.image_size,d=d)[:,None]

        if round:
            fx = np.fft.rfftfreq(self.image_size,d=d)[: (self.image_size // 2) + 1]
        else:
            fx = np.fft.rfftfreq(self.image_size,d=d)[: self.image_size // 2]

        freqs = (fx*fx + fy*fy) ** decay_power
        scale = 1.0 / np.maximum(freqs, 1.0 / (self.image_size*d))
        scale = torch.tensor(scale).float().to(self.device)

        return scale

    def fft_to_rgb(self, decay_power):
        """convert image param to NCHW 
        WARNING: torch v1.7.0 works differently from torch v1.8.0 on fft. 
        torch-dreams supports ONLY 1.8.x 
        Latest docs: https://pytorch.org/docs/stable/fft.html
        Also refer:
            https://github.com/pytorch/pytorch/issues/49637
        Args:
            height (int): height of image
            width (int): width of image 
            image_parameter (auto_image_param): auto_image_param.param
        Returns:
            torch.tensor: NCHW tensor
        """
        scale = self.get_fft_scale(decay_power).to(self.device)
        # print(scale.shape, image_parameter.shape)
        if self.image_size %2 ==1:
            image_parameter = self.spectrum_t.reshape(-1,3,self.image_size, (self.image_size+1)//2, 2)
        else:
            image_parameter = self.spectrum_t.reshape(-1,3,self.image_size, self.image_size//2, 2)

        image_parameter = torch.complex(image_parameter[..., 0], image_parameter[..., 1])
        t = scale * image_parameter

        version = torch.__version__.split('.')[:2]
        main_version = int(version[0])
        sub_version = int(version[1])

        if  main_version >= 1 and sub_version >= 8:  ## if torch.__version__ is greater than 1.8
            t = torch.fft.irfft2(t,  s = (self.image_size, self.image_size), norm = 'ortho')
        else:
            raise PytorchVersionError(version = torch.__version__)

        return t

class PytorchVersionError(Exception):
    """Raised when the user is not on pytorch 1.8.x or higher
    Args:
        version (str): torch.__version__
    """ 
    def __init__(self, version):
        self.version = version
        self.message = "Expected pytorch to have version 1.8.x or higher but got: " + self.version + "\n Please consider updating pytorch from: https://pytorch.org/get-started/locally"
    def __str__(self):
        return self.message