import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def load_vision_model(model_name, device=None, ckpt_path=None):
    if model_name=="vit-base":
        model_name = "vit_base_patch16_224"

    if model_name=="vit-base.dino":
        model_name = "vit_base_patch16_224_dino"

    if ckpt_path is not None:
        model = torch.load(ckpt_path, map_location=device)
    else:
        model = timm.create_model(model_name, pretrained=True)
    
    if device is not None:
        model = model.to(device)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    if "resnet" in model_name:
        pool_func = model.global_pool
        decision_layer = model.fc
    elif "vit" in model_name:
        pool_func = lambda x: model.fc_norm(x[:,0])
        decision_layer = model.head
    elif "efficientnet" in model_name:
        pool_func = model.global_pool
        decision_layer = model.classifier
    elif "vgg" in model_name:
        pool_func = model.pre_logits
        decision_layer = model.head

    return transform, model, pool_func, decision_layer, config