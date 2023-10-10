# Simple script to demonstrate CCE
import os
import pickle
import argparse
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import MetashiftManager

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument('--dataset', type=str, default="bear-bird-cat-dog-elephant:dog(water)")
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument("--ckpt_dir", type=str, default="ckpts/bear-bird-cat-dog-elephant:dog(snow)")
    
    parser.add_argument('--model-path', type=str, default=None, 
                        help="If provided, will use this model instead of the one in the out_dir")
    
    parser.add_argument("--num-mistakes", default=50, type=int, help="Number of mistakes to evaluate")
    
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    
    ## CCE parameters
    parser.add_argument("--step-size", default=1e-2, type=int, help="Stepsize for CCE")
    parser.add_argument("--alpha", default=1e-1, type=float, help="Alpha parameter for CCE")
    parser.add_argument("--beta", default=1e-2, type=float, help="Beta parameter for CCE")
    parser.add_argument("--n_steps", default=100, type=int, help="Number of steps for CCE")

    return parser.parse_args()


    
def main(args):
    sns.set_context("poster")
    np.random.seed(args.seed)

    model = timm.create_model(args.model_name, pretrained=False)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    if args.model_path is None:
        model_path = os.path.join(args.ckpt_dir, "confounded-model.pt")
    else:
        model_path = args.model_path

    # Load the model
    model = torch.load(model_path)
    model = model.to(args.device)
    model = model.eval()
    
    # Get the images that do not contain the spurious concept
    manager = MetashiftManager()
    train_domain = args.dataset.split(":")[1]
    shift_class = train_domain.split("(")[0]
    spurious_concept = train_domain.split("(")[1][:-1].lower()
    print(f"Shift Class: {shift_class}, Spurious Concept: {spurious_concept}")
    
    shift_class_images = manager.get_single_class(shift_class, exclude=train_domain)
    np.random.shuffle(shift_class_images)
    
    cls_idx = args.dataset.split(":")[0].split("-").index(shift_class)
    
    num_eval = 0
    tqdm_iterator = tqdm(shift_class_images)
    for image_path in tqdm_iterator:
        # Read the image and label
        image = Image.open(image_path)
        
        with torch.no_grad():
            image_tensor = transform(image).unsqueeze(0).to(args.device)
            label = cls_idx*torch.ones(1, dtype=torch.long).to(args.device)
            
            # Get the model prediction
            pred = model(image_tensor)
            pred = pred.argmax(dim=1)
            
            # Only evaluate over mistakes
            if pred.item() == label.item():
                continue
            
            num_eval += 1
        
        if num_eval == args.num_mistakes:
            break

    print(num_eval)
    
if __name__ == "__main__":
    args = config()
    main(args)