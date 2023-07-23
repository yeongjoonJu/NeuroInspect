import os, time
from typing import Callable, List
from tqdm import tqdm
from glob import glob

from utils.hook import Hook

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.dir_handler as dh
from utils.params import ImageParams
from utils.prompts import prepare_class_names
from utils.config import revised_prompt_templates

# import clip
import open_clip


class Illusion(object):
    def __init__(self, model:nn.Module, decision:nn.Module, objective_fn, \
        class_dict, save_dir="results", device="cuda:0", \
        img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225], ):

        self.device = torch.device(device)
        self.save_dir = save_dir
        dh.make_folder_if_it_doesnt_exist(name=save_dir)
        self.class_dict = class_dict

        self.model = model.to(self.device)
        self.model.eval()
        self.decision = decision
        self.objective_fn = objective_fn
    
        # self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained='laion2b_s34b_b79k')
        self.clip_model.to(self.device)
        self.clip_model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
            
        self.image_mean = torch.tensor([img_mean]).unsqueeze(-1).unsqueeze(-1).to(self.device)
        self.image_std = torch.tensor([img_std]).unsqueeze(-1).unsqueeze(-1).to(self.device)
        self.clip_mean=torch.tensor((0.48145466, 0.4578275, 0.40821073)).unsqueeze(-1).unsqueeze(-1).to(self.device)
        self.clip_std =torch.tensor((0.26862954, 0.26130258, 0.27577711)).unsqueeze(-1).unsqueeze(-1).to(self.device)
        
    def clip_process(self, image):
        return (image - self.clip_mean) / self.clip_std

    def postprocess(self, image):
        return (image - self.image_mean) / self.image_std
    
    def optimize_image(self, layer, params, objective, iters, lr, weight_decay=1e-2, class_indices=None, quiet=True):
        params.get_optimizer_and_scheduler(param_list=params.get_param_list(), lr=lr, weight_decay=weight_decay, eta_min=lr*0.1)

        hook = Hook(layer)
        channel_number = objective.channel_number
        if class_indices is not None:
            objective.class_idx = class_indices

        tqdm_obj = tqdm(range(iters), disable=quiet, total=iters)
        for t in tqdm_obj:
            params.optimizer.zero_grad()

            image = params()
            model_out = self.model(self.postprocess(image))
            layer_out = hook.output
            if class_indices is None:
                loss = objective(image, layer_out)
            else:
                # class_idx = model_out.argmax(dim=-1)[0].item()
                loss = objective(image, layer_out, model_out)

            tqdm_obj.set_postfix(loss=loss.item(), lr=params.get_lr())

            loss.backward()
            params.optimizer.step()
            params.lr_scheduler.step(loss.item())

        hook.close()

        image = params.to_chw_tensor().detach()
        self.model(self.postprocess(image))
        layer_out = hook.output
        activations = objective.activation(layer_out)
        hook.close()
        image = image.detach()

        objective.channel_number = channel_number

        return image, activations


    def dreaming(self, layer, objective_fn, image_size, iters, class_indices=None, \
                img_param=None, batch_size=4, lr=0.015, weight_decay=1e-2, init_image=None, \
                scale_min=0.7, scale_max=1.2, rotate_degrees=15, translate_x=0.15, translate_y=0.15, color_aug=False):
        # dreaming
        if img_param is None:
            img_param = ImageParams(image_size=image_size, batch_size=batch_size, device=self.device, max_iter=iters, init_image=init_image, \
                scale_min=scale_min, scale_max=scale_max, rotate_degrees=rotate_degrees, translate_x=translate_x, translate_y=translate_y, color_aug=color_aug)
        _, act = self.optimize_image(layer, img_param, objective_fn, iters, lr, class_indices=class_indices, weight_decay=weight_decay, quiet=True)
        # max_act_idx = torch.argmax(act, dim=0)
        dream = img_param.to_chw_tensor().detach()
        
        return dream, act.detach()
    

    def get_class_text_embedding(self, text, format=True):
        if format:
            prompts = [temp.format(text) for temp in revised_prompt_templates]
        else:
            prompts = [f"a close up of a person, {text}, associated press photo"]
        target_tokenized = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            target_text_feats = self.clip_model.encode_text(target_tokenized)
            target_text_feats = target_text_feats / target_text_feats.norm(dim=-1, keepdim=True)
            target_text_feats = target_text_feats.mean(dim=0)
            # target_text_feats /= target_text_feats.norm()
        
        return target_text_feats.unsqueeze(0)

    
    def optimize_caption_and_dream(self, layer, color_aug=False, init_image=None, thresholding=True, \
                                lr=9e-3, weight_decay=1e-3, iters=1024, texts=["image"],\
                                quiet=False, threshold=0.5, reduction=0.5, batch_size=1):

        image_size = self.objective_fn.image_size
        img_param = ImageParams(init_image=init_image, image_size=image_size, batch_size=batch_size, device=self.device, max_iter=iters, \
                            scale_min=0.6, scale_max=1.2, rotate_degrees=15, translate_x=0.15, translate_y=0.15, color_aug=False)
    
        hook = Hook(layer)
        img_param.get_optimizer_and_scheduler(param_list=img_param.get_param_list(), lr=lr, weight_decay=weight_decay, warmup=False, eta_min=0.12*lr)

        if len(texts) > 1:
            target_text_feats = [self.get_class_text_embedding(text, format=True) for text in texts] #  
            target_text_feats = torch.cat(target_text_feats)
        else:
            target_text_feats = self.get_class_text_embedding(texts[0], format=True) # 
        
        # Visualization
        tqdm_obj = tqdm(range(iters), disable=quiet, total=iters)
        for t in tqdm_obj:
            img_param.optimizer.zero_grad()

            image = img_param(training=True)
            
            # Encode image embedding using CLIP
            img_feats = self.clip_model.encode_image(self.clip_process(image))
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            
            logits = self.model(self.postprocess(image))
            layer_out = hook.output

            if color_aug:
                layer_out = layer_out[:, 0]
            else:
                B,C,_,_ = layer_out.shape
                layer_out = layer_out.view(B,C,-1).mean(dim=-1)
            
            loss = self.objective_fn(layer_out, logits, \
                                    img_feats, target_text_feats)
            
            
            tqdm_obj.set_postfix(loss=loss.item(), lr=img_param.get_lr())
            
            loss.backward()
            img_param.optimizer.step()
            img_param.lr_scheduler.step()

        image = img_param.to_chw_tensor().detach()
        self.model(self.postprocess(image))
        layer_out = hook.output
        activations = self.objective_fn.activation(layer_out)
        hook.close()
        image = image.detach()

        if self.objective_fn.is_vit:
            hook = Hook(self.model.blocks[-2])
            self.model(self.postprocess(image))
            layer_out = hook.output

        if thresholding:
            masks = self.objective_fn.activation_map(layer_out.detach().clone(), reduction=reduction, threshold=threshold)
            image = image*masks
        else:
            masks = self.objective_fn.activation_map(layer_out.detach().clone(), reduction=0.0, threshold=0.1)

        return image, activations, masks
    
    def set_one_neuron_multiple_classes_in_batch(
        self,
        target_neuron,
        batch_size=1,
        class_indices=None, #  List[int]
    ):
        if class_indices is not None:
            batch_size = len(class_indices)
            
        # set neuron idx
        target_neuron = torch.LongTensor([target_neuron for _ in range(batch_size)]).to(self.device)
        self.objective_fn.channel_number = target_neuron
        
        # set class idx
        if class_indices is None:
            # Class most related to the neuron
            dummy = torch.zeros(batch_size, self.decision.weight.data.shape[1])
            dummy[:,target_neuron] = 30.0
            logits = self.decision(dummy.to(self.device))
            class_indices = torch.topk(logits[0], k=batch_size, dim=0)[1]
        else:
            class_indices = torch.LongTensor(class_indices).to(self.device)
        
        self.objective_fn.class_idx = class_indices
        
        # Generate Feature
        class_names = prepare_class_names(class_indices, self.class_dict)
        
        return class_names
    
    
    def set_multiple_neurons_one_class_in_batch(
        self,
        class_idx,
        batch_size=1,
        target_neurons=None, # List or LongTensor
    ):
        if target_neurons is not None:
            batch_size = len(target_neurons)
            
        # set neuron idx
        if target_neurons is None:
            target_neurons = torch.topk(self.decision.weight.data[class_idx], k=batch_size, dim=0, largest=True)[1]
        
        self.objective_fn.channel_number = target_neurons.to(self.device)
        
        # set class idx
        class_indices = torch.LongTensor([class_idx for _ in range(batch_size)]).to(self.device)
        self.objective_fn.class_idx = class_indices
        
        # Generate Feature
        class_names = prepare_class_names(class_indices, self.class_dict)
        
        return class_names
        
    
    def visualize_neurons(
        self,
        experiment_name,
        layer: nn.Module,
        target_neurons: List[int],
        iters=150,
        lr=1.5e-2,
        batch_size=1,
        weight_decay=0.0,
        overwrite_experiment=True,
        color_aug=False,
        quiet=False,
        thresholding=True,
        class_idx=None,
        out_path=False,
        reduction=0.5,
        threshold=0.5,
    ):
        """Class-oriented neuron visualization
        Conditioning on a class label most activated by a neuron to be investigated.
        """
         # time when execution started
        starting_time = time.time()
        experiment_folder = dh.set_experiment_dir(self.save_dir, experiment_name, overwrite_experiment, starting_time, "illusion")
        
        if type(class_idx) is int:
            class_idx = [class_idx for _ in range(batch_size)]
        
        viz_out = []
        act_out = []
        mask_out = []
        for neuron_idx in tqdm(target_neurons, desc="Visualizing Neurons"):
            dir_name = f"{experiment_folder}/{neuron_idx}"
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
                
            # one neuron and one class in batch    
            class_names = self.set_one_neuron_multiple_classes_in_batch(\
                neuron_idx, batch_size, class_indices=class_idx)
            
            # Generate CLIP-Illusion
            images, acts, masks = self.optimize_caption_and_dream(layer, batch_size=batch_size, color_aug=color_aug, reduction=reduction, \
                                                        threshold=threshold, thresholding=thresholding, lr=lr, weight_decay=weight_decay, \
                                                        iters=iters, texts=class_names, quiet=quiet)
            
            dh.save_illusion_results(images if thresholding else images*masks, acts, dir_name, neuron_idx, class_name=" | ".join(class_names))
            
            viz_out.append(images.detach())
            act_out.append(acts.squeeze(-1).detach())
            mask_out.append(masks.detach())
            
        viz_out = torch.cat(viz_out)
        act_out = torch.cat(act_out)
        mask_out = torch.cat(mask_out)
        
        if out_path:
            preds = self.model(self.postprocess(viz_out))
            preds = preds.argmax(dim=-1)
            class_mask = torch.eq(preds, class_idx[0])
            paths = []
            for neuron_idx in target_neurons:
                dir_name = f"{experiment_folder}/{neuron_idx}"
                paths.extend(glob(f"{dir_name}/*.jpg"))
            
            viz_paths = []
            for n in range(class_mask.shape[0]):
                if class_mask[n]:
                    viz_paths.append(paths[n])
            
            return viz_paths
        
        return viz_out, act_out, mask_out