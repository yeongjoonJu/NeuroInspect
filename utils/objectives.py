import torch
import numpy as np
import torch.nn.functional as F
import math

class ImageLoss:
    def __init__(self, coefficient, image_size):
        self.c = coefficient
        self.image_size = image_size
    
    def __call__(self, x: torch.tensor):
        loss = self.loss(x)
        return self.c * loss
    
    def loss(self, x: torch.tensor):
        raise NotImplementedError

class ChannelLoss:
    def __init__(self, coefficient=1, channel_number=0):
        self.c = coefficient
        self.channel_number = channel_number
    
    def __call__(self, x: torch.tensor):
        loss = self.loss(x)
        return self.c * loss
    
    def loss(self, x: torch.tensor):
        raise NotImplementedError


class TotalVariance(ImageLoss):
    def __init__(self, coefficient=1, image_size=224, p=2):
        super().__init__(coefficient=coefficient, image_size=image_size)
        self.p = p

    def total_variation(self, x):
        x_wise = x[:, :, :, 1:] - x[:, :, :, :-1]
        y_wise = x[:, :, 1:, :] - x[:, :, :-1, :]
        diag_1 = x[:, :, 1:, 1:] - x[:, :, :-1, :-1]
        diag_2 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
        return x_wise.norm(p=self.p, dim=(2, 3)).mean() + y_wise.norm(p=self.p, dim=(2, 3)).mean() + \
               diag_1.norm(p=self.p, dim=(2, 3)).mean() + diag_2.norm(p=self.p, dim=(2, 3)).mean()
    
    def loss(self, x:torch.tensor):
        tv = self.total_variation(x)
        return tv * np.prod(x.shape[-2:]) / self.image_size

class Diversity(ChannelLoss):
    def diversity(self, layer_out):
        if len(layer_out.shape)>=3:
            layer_out = layer_out.view(layer_out.shape[0], layer_out.shape[1], -1)
            layer_out = layer_out.mean(dim=-1)
            
        b = layer_out.shape[0]
        flattened = layer_out.view(b, -1)
        gram = flattened @ flattened.t()
        gram = gram / gram.norm(p=2)
        gram = torch.triu(gram, diagonal=1)
        return gram.sum(1) # should reduce
    
    def loss(self, x: torch.tensor):
        return self.diversity(x).mean()
    
    
class ChannelObjective(torch.nn.Module):
    def __init__(self, channel_number=0, class_idx=0, image_size=224, is_vit=False): 
        self.is_vit = is_vit
        self.channel_number = channel_number
        self.class_idx = class_idx
        self.image_size = image_size

    def forward(self, layer_out, decision_out):
        loss = self.activation(layer_out)
        if type(self.class_idx) is int:
            logits = decision_out[:,self.class_idx]
        else:
            logits = torch.gather(decision_out, 1, self.class_idx.view(-1,1))
        
        return -1*(loss + logits*0.1)
    
    def __call__(self, layer_out, decision_out):
        loss = self.forward(layer_out, decision_out)
        return loss.mean()
    
    def activation(self, layer_out):
        if len(layer_out.shape) > 2:
            channel_out = layer_out.view(layer_out.shape[0], layer_out.shape[1], -1)
            channel_out = channel_out.mean(dim=-1)
        else:
            channel_out = layer_out
        
        
        if type(self.channel_number) is int:
            channel_out = channel_out[:,self.channel_number]
        else:
            channel_out = torch.gather(channel_out, 1, self.channel_number.view(-1,1))
        
        return channel_out
    
    def activation_map(self, layer_out, threshold=0.0, reduction=0.25):
        if self.is_vit:
            layer_out = layer_out[:,1:]
            layer_out = layer_out.permute(0,2,1) # B D S
            hw = int(math.sqrt(layer_out.shape[2]))
            layer_out = layer_out.view(-1,layer_out.shape[1], hw, hw)
        
        if type(self.channel_number) is int:
            layer_out = layer_out[:,self.channel_number] # B H W
        else:
            B, _, H, W = layer_out.shape
            layer_out = torch.gather(layer_out, dim=1, index=self.channel_number.view(-1,1,1,1).expand(-1,-1,H,W))
            layer_out = layer_out.squeeze(1)
        
        B = layer_out.size(0)
        
        mask = layer_out.unsqueeze(1)
        mask = F.interpolate(mask, (self.image_size, self.image_size), mode="bilinear", align_corners=True)
        # ori_mask = mask.clone()
        
        flatten = mask.view(B,-1)
        mean = flatten.mean(dim=1).view(B,1,1,1)
        std = flatten.std(dim=1).view(B,1,1,1)
        stat = (mask - mean) / std
        
        mask[stat < threshold] = 0.0
        mask[mask > 0.0] = 1.0
        mask[mask==0.0] = reduction

        return mask.clamp(0.0,1.0)


class ClassConditionalObjective(ChannelObjective):
    def __init__(self, channel_number=0, class_idx=0, image_size=224, is_vit=False,\
                class_gamma=0.5, domain_eps=0.01,): 
        super().__init__(channel_number, class_idx, image_size, is_vit)
        self.class_gamma = class_gamma
        self.domain_eps = domain_eps
        
    def clip_loss(self, img_feats, text_feats):
        loss_clip = img_feats @ text_feats.detach().t()
        loss_clip = torch.diag(loss_clip)
        
        return loss_clip

    def forward(self, layer_out, decision_out):
        loss = self.activation(layer_out)
        if type(self.class_idx) is int:
            logits = decision_out[:,self.class_idx]
        else:
            logits = torch.gather(decision_out, 1, self.class_idx.view(-1,1))
            
        class_adj = self.class_gamma#loss.clamp(min=0.01, max=self.class_gamma)
            
        return -1*(loss + logits*class_adj)
    
    def __call__(self, layer_out, decision_out, img_feats, text_feats):
        loss = self.forward(layer_out, decision_out)
        loss_clip = self.clip_loss(img_feats, text_feats)
        loss_ce = F.cross_entropy(decision_out, self.class_idx, reduction="none")
        
        loss = (loss_clip+self.domain_eps)*loss
        
        return (loss + loss_ce).mean()
        # return loss.mean()