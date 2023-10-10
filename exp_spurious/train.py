# A simple training script for metashift scenarios.

import argparse
import random
import sys
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from tqdm import tqdm
from dataset import load_data

import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def config():
    parser = argparse.ArgumentParser()
    
    # Model and data details
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='bear-bird-cat-dog-elephant:dog(water)')
    
    # Training details
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    # File details
    parser.add_argument('--out_dir', type=str, default='ckpts')
   
    return parser.parse_args()


@torch.no_grad()
def eval_loop(loader, model, device):
    model.eval()
    accuracymeter = AverageMeter()
    lossmeter = AverageMeter()
    criterion = nn.CrossEntropyLoss() 
    tqdm_loader = tqdm(loader)
    for inputs, labels in tqdm_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        preds = torch.argmax(outputs, 1)
        
        # Log the accuracy and the loss
        accuracymeter.update((preds == labels).float().mean().cpu().numpy(), inputs.size(0))
        lossmeter.update((preds == labels).float().mean().cpu().numpy(), inputs.size(0))
        tqdm_loader.set_postfix(Acc=accuracymeter.avg, Loss=lossmeter.avg)
    
    return accuracymeter.avg
    


def train_model(train_loader, val_loader, model, optimizer, scheduler, num_epochs, device, save_dir):    
    criterion = nn.CrossEntropyLoss() 

    best_val_acc = 0.0    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()
  
        tqdm_loader = tqdm(train_loader)
        for inputs, labels in tqdm_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            tqdm_loader.set_postfix(loss=loss.item())

        model.eval()
        print('Evaluating on the training set...')
        eval_loop(train_loader, model, device)
        print('Evaluating on validation set...')
        val_acc = eval_loop(val_loader, model, device)
        if val_acc > 0.7 and val_acc > best_val_acc:
            best_val_acc = val_acc

            # Save the model and configurations
            # model = model.to("cpu")
            print("Save checkpoint", val_acc)
            torch.save(model, os.path.join(save_dir, "confounded-model_best.pt"))
            # model = model.to(device)


def main(args):
    # model, _, _, train_preprocess, val_preprocess = get_model(args, get_full_model=True, eval_mode=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = timm.create_model(args.model_name, pretrained=True).to(args.device)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    loaders, cls_to_lbl, data_meta_info = load_data(args, transform, transform, args.dataset)

    # Change the classifier layer
    model.fc = nn.Linear(model.fc.in_features, len(cls_to_lbl))
    for p in model.fc.parameters():
        p.requires_grad = True
    
    model = model.to(args.device)
    
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    max_iter = len(loaders["train"])*args.num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=2e-4)

    scenario_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(scenario_dir, exist_ok=True)

    # Train the model
    train_model(loaders["train"], loaders["val"], model, optimizer, scheduler, \
                args.num_epochs, args.device, save_dir=scenario_dir)

    with open(os.path.join(scenario_dir, "data_meta_info.pkl"), "wb") as f:
            pickle.dump(data_meta_info, f)
    with open(os.path.join(scenario_dir, "args"), "wb") as f:
            pickle.dump(args, f)
    
    
if __name__ == "__main__":
    args = config()
    main(args)