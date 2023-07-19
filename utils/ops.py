import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np


def resize_4d_tensor_by_factor(x, height_factor, width_factor):
    res = F.interpolate(x, scale_factor= (height_factor, width_factor), mode = 'bilinear')
    return res

def resize_4d_tensor_by_size(x, height, width):
    res = F.interpolate(x, size =  (height, width), mode = 'bilinear')
    return res


class random_resize(nn.Module):
    def __init__(self, max_size_factor, min_size_factor):
        super().__init__()
        self.max_size_factor = max_size_factor
        self.min_size_factor = min_size_factor

    def forward(self, x):
    
        # size = random.randint(a = 300, b = 600)
        # resized= resize_4d_tensor_by_size(x = x, height = size, width = size)

        height_factor = random.uniform(a = self.min_size_factor , b = self.max_size_factor)
        width_factor = random.uniform(a = self.min_size_factor , b = self.max_size_factor)

        resized = resize_4d_tensor_by_factor(x = x, height_factor = height_factor, width_factor = width_factor)
        return resized

class GaussianNoise(nn.Module):
    def __init__(self, batch_size: int, shuffle_every: bool = False, std: float = 1.0, max_iter: int = 400, device=0):
        super().__init__()
        self.batch_size, self.std_p, self.max_iter = batch_size, std, max_iter
        self.std = None
        self.rem = max_iter - 1
        self.device = device
        self.shuffle(device)
        self.shuffle_every = shuffle_every
        

    def shuffle(self, device):
        self.std = torch.randn(self.batch_size, 3, 1, 1).to(device) * self.rem * self.std_p / self.max_iter
        self.rem = (self.rem - 1 + self.max_iter) % self.max_iter

    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle(self.device)
        return img + self.std

class Jitter(nn.Module):
    def __init__(self, lim: int = 32):
        super().__init__()
        self.lim = lim

    def forward(self, x: torch.tensor) -> torch.tensor:
        off1 = random.randint(-self.lim, self.lim)
        off2 = random.randint(-self.lim, self.lim)
        return torch.roll(x, shifts=(off1, off2), dims=(2, 3))

class ColorJitter(nn.Module):
    def __init__(self, batch_size: int, shuffle_every: bool = False, mean: float = 1., std: float = 1., device=0):
        super().__init__()
        self.batch_size, self.mean_p, self.std_p = batch_size, mean, std
        self.mean = self.std = None
        self.device = device
        self.shuffle(device)
        self.shuffle_every = shuffle_every

    def shuffle(self, device):
        self.mean = (torch.rand((self.batch_size, 3, 1, 1,)).to(device) - 0.5) * 2 * self.mean_p
        self.std = ((torch.rand((self.batch_size, 3, 1, 1,)).to(device) - 0.5) * 2 * self.std_p).exp()

    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle(self.device)
        return (img - self.mean) / self.std


def get_transforms(image_size, rotate_degrees=15, scale_max=1.2, scale_min=0.5, \
                translate_x = 0.15, translate_y = 0.15):
    transform= transforms.Compose([
        transforms.RandomAffine(degrees = rotate_degrees, translate= (translate_x, translate_y)),
        random_resize(max_size_factor = scale_max, min_size_factor = scale_min),
        transforms.CenterCrop((image_size, image_size))
    ])

    return transform

def color_correlation_normalized():
    color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                             [0.27, 0.00, -0.05],
                                             [0.27, -0.09, 0.03]]).astype(np.float32)
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
    color_correlation_normalized = torch.tensor(color_correlation_svd_sqrt / max_norm_svd_sqrt)
    return color_correlation_normalized

def imagenet_mean_std():
    return (torch.tensor([0.485, 0.456, 0.406]), 
            torch.tensor([0.229, 0.224, 0.225]))

class Constants:
    color_correlation_matrix = color_correlation_normalized()
    imagenet_mean, imagenet_std = imagenet_mean_std()


def lucid_colorspace_to_rgb(t, device):
    t_flat = t.permute(0,2,3,1)
    t_flat = torch.matmul(t_flat.to(device) , Constants.color_correlation_matrix.T.to(device))
    t = t_flat.permute(0,3,1,2)

    return t

def rgb_to_lucid_colorspace(t, device):
    t_flat = t.permute(0, 2, 3, 1)
    inverse = torch.inverse(Constants.color_correlation_matrix.T.to(device))
    t_flat = torch.matmul(t_flat.to(device), inverse)
    t = t_flat.permute(0, 3, 1, 2)

    return t

def bchw_rgb_to_fft_param(x):
    im_tensor = x.float()

    x = rgb_to_lucid_colorspace(denormalize(im_tensor), device=x.device)
    x = torch.fft.rfft2(x, s=(x.shape[-2], x.shape[-1]), norm="ortho")
    return x

def normalize_image(x, device):
    return (x-Constants.imagenet_mean[...,None,None].to(device)) / Constants.imagenet_std[...,None,None].to(device)

def denormalize(x):
    return x.float() * Constants.imagenet_std[..., None, None].to(
        x.device
    ) + Constants.imagenet_mean[..., None, None].to(x.device)

def to_rgb(x):
    x = x.permute(0,2,3,1).numpy()
    x = (x*255.0).astype(np.uint8)
    return x


def cosine_similarity2D(vectors):
    """
    vectors: [S X D]
    """
    norm_v = torch.linalg.norm(vectors, dim=-1) # S
    vectors = vectors.t() # D X S
    _vectors = vectors.unsqueeze(1)
    vectors = vectors.unsqueeze(-1)
    sim_map = torch.matmul(vectors, _vectors) # D X S X S
    sim_map = torch.sum(sim_map, dim=0)
    norm_v = torch.matmul(norm_v.unsqueeze(0), norm_v.unsqueeze(-1))
    norm_v = torch.clamp(norm_v, min=1e-8)
    sim_map = sim_map / norm_v # S X S

    return sim_map

def top_k_top_p_filtering(logit, top_k=0, top_p=1.0, filter_value=-float("Inf")):

    if top_k > 0:
        top_k = min(max(top_k, 1), logit.size(-1)) # sanity check
        # remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logit < torch.topk(logit, top_k)[0][...,-1,None]
        logit[indices_to_remove] = filter_value            
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logit, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,1:] = sorted_indices_to_remove[...,:-1].clone()
        sorted_indices_to_remove[...,0] = 0
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logit[indices_to_remove] = filter_value

    return logit

def cosine_dissimilarity(x, y, eps=1e-6, power=1.0):
    numerator = (x * y.detach()).sum()
    denominator = torch.sqrt((y**2).sum()) + eps
    cossim = numerator / denominator
    cossim = torch.maximum(torch.tensor(0.1).to(cossim.device), cossim)
    
    return -1*cossim * numerator**power


def blur(image, filter_size=5):
    """
    Args : Tensor N x H x W or N x C x H x W
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(1)

    channels = image.shape[1]
    kernel = torch.ones(1, 1, filter_size, filter_size) / (filter_size*filter_size)

    out = None
    padding = (filter_size-1)//2
    for channel in range(channels):
        _out = F.conv2d(image[:,channel,...].unsqueeze(1), kernel.to(image.get_device()), padding=padding)
        if out is None:
            out = _out
        else:
            out = torch.cat([out, _out], dim=1)
    
    return out