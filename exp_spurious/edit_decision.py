# A simple training script for metashift scenarios.

import argparse
import random
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import ResNet

from tqdm import tqdm
from dataset import load_data


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
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument("--save_name", type=str, default="edited")
    
    # Training details
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16) # 16
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--neurons", nargs="+", type=int, required=True)
    parser.add_argument("--modifying_class", type=int, required=True)
    
    # File details
    parser.add_argument('--out_dir', type=str, default='ckpts')
   
    return parser.parse_args()


@torch.no_grad()
def eval_loop(loader, model, device, neuron_mask, mod_class):
    model.eval()
    accuracymeter = AverageMeter()
    lossmeter = AverageMeter()
    avg_l2 = 0.0

    tqdm_loader = tqdm(loader)
    for inputs, labels in tqdm_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        features = model.forward_features(inputs)
        features = model.global_pool(features)
        outputs = model.fc(features)
        features_masked = features*neuron_mask.detach()
        
        prob = torch.softmax(outputs, dim=1)
        changed_prob = torch.softmax(model.fc(features_masked), dim=1)
        retaining_ratio = (prob / changed_prob)
        l2_loss = torch.norm(retaining_ratio[:,mod_class]-1.0, dim=0, p=2).mean()
        avg_l2 = avg_l2 + l2_loss.item()


        preds = torch.argmax(outputs, 1)
        
        # Log the accuracy and the loss
        accuracymeter.update((preds == labels).float().mean().cpu().numpy(), inputs.size(0))
        lossmeter.update((preds == labels).float().mean().cpu().numpy(), inputs.size(0))
        tqdm_loader.set_postfix(Acc=accuracymeter.avg, Loss=lossmeter.avg)
    
    return accuracymeter.avg, avg_l2 / len(loader)


def get_warmup_lr_scheduler(optimizer, init_lr, lr_rampdown_length=0.25, lr_rampup_length=0.05):
    def lr_scheduler(step, num_steps):
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = init_lr * lr_ramp
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    
    return lr_scheduler


def train_model(train_loader, val_loader, model, optimizer, scheduler, num_epochs, device, save_dir,\
                save_name, mod_class=None, neuron_mask=None):
    
    criterion = nn.CrossEntropyLoss()

    max_iter = len(train_loader)*num_epochs
    best_val_acc = 0.0
    best_val_reg = 99

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()

        tqdm_loader = tqdm(enumerate(train_loader))
        for t, (inputs, labels) in tqdm_loader:
            
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            features = model.forward_features(inputs)
            features = model.global_pool(features)
            outputs = model.fc(features)
            features_masked = features*neuron_mask.detach()

            ce_loss = criterion(outputs, labels)

            if False:
                neuron_mask = 1. - neuron_mask
                contrib = model.fc.weight[mod_class].unsqueeze(0)*features + model.fc.bias[mod_class]
                contrib = contrib.clamp(min=0.0)
                contrib = contrib * neuron_mask
                l2_loss = torch.norm(contrib, dim=1, p=2).mean()
            else:
                prob = torch.softmax(outputs, dim=1)
                changed_prob = torch.softmax(model.fc(features_masked), dim=1)
                retaining_ratio = (prob / (changed_prob+1e-15))
                
                l2_loss = torch.norm((retaining_ratio[:,mod_class]-1.0), dim=0, p=2)
                l2_loss = l2_loss.mean()
                
            loss = ce_loss + l2_loss
            
            tqdm_loader.set_postfix(ce=ce_loss.item(), l2=l2_loss.item())
            loss.backward()
            optimizer.step()
            scheduler(t, max_iter)
            
        model.eval()

        print('Evaluating on validation set...')
        val_acc, val_reg = eval_loop(val_loader, model, device, neuron_mask, mod_class)
        print("Val acc, val_reg", val_acc, val_reg)
        
        if val_acc > 0.7 and epoch > 1 and best_val_acc <= val_acc:
            if best_val_acc==val_acc and best_val_reg < val_reg:
                continue
            
            best_val_acc = val_acc

            # Save the model and configurations
            print("Save checkpoint", val_acc)
            torch.save(model, os.path.join(save_dir, f"{save_name}.pt"))

"""
python edit_decision.py --dataset="bear-bird-cat-dog-elephant:dog(water)" --device 0 --modifying_class 3 --neurons 510 
python edit_decision.py --dataset="bear-bird-cat-dog-elephant:dog(snow)" --device 1 --modifying_class 3 --neurons 166 283 
"""

def main(args):
    os.environ['PYTHONHASHSEED']=str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ckpt_path = f"ckpts/{args.dataset}/confounded-model_best.pt"
    model = torch.load(ckpt_path, map_location=f"cuda:{args.device}")
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    loaders, cls_to_lbl, data_meta_info = load_data(args, transform, transform, args.dataset)
    
    num_classes = len(cls_to_lbl)
    # Change the classifier layer
    # model.fc = nn.Linear(model.fc.in_features, len(cls_to_lbl))
    for p in model.fc.parameters():
        p.requires_grad = True
    
    model = model.to(args.device)
    
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    scheduler = get_warmup_lr_scheduler(optimizer, args.lr)

    scenario_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(scenario_dir, exist_ok=True)

    print("Save dir:", scenario_dir)

    embed_dim= model.fc.weight.data.shape[1]
    index = torch.LongTensor(args.neurons)
    neuron_mask = torch.ones(embed_dim)
    neuron_mask.index_fill_(0, index, 0)
    neuron_mask = neuron_mask.unsqueeze(0).to(args.device)

    # Train the model
    train_model(loaders["train"], loaders["val"], model, optimizer, scheduler, \
                args.num_epochs, args.device, save_dir=scenario_dir, save_name=args.save_name, \
                mod_class=args.modifying_class, neuron_mask=neuron_mask)

    with open(os.path.join(scenario_dir, "data_meta_info.pkl"), "wb") as f:
            pickle.dump(data_meta_info, f)
    with open(os.path.join(scenario_dir, "args"), "wb") as f:
            pickle.dump(args, f)
    
    
if __name__ == "__main__":
    args = config()
    main(args)