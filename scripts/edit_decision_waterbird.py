# Author: merty
# A simple training script for metashift scenarios.

import argparse
import numpy as np
import os, json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from tqdm import tqdm
from train_test_classifier import load_datasets, load_test_data, set_seed, eval_loop

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
        
@torch.no_grad()
def eval_loop_group(loader, model, device):
    model.eval()
    tqdm_loader = tqdm(enumerate(loader), desc="Evaluating")
    
    # Initialize dictionary for storing class-wise accuracy and count
    group_correct = [0 for _ in range(4)]
    group_total = [0 for _ in range(4)]

    avg_correct = 0
    avg_total = 0

    class_correct = {}
    class_total = {}

    for idx, batch in tqdm_loader:
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        g_labels = batch[2].to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, 1)
        
        # Update overall accuracy and loss meters
        corrected = (preds==labels).float()
        avg_correct += corrected.sum().item()
        avg_total += len(labels)
        for g in range(4):
            group_correct[g] += (corrected[g_labels==g]).sum().item()
            group_total[g] += (g_labels==g).sum().item()

        for label in list(set(batch[1].tolist())):
            label_mask = labels==label
            if label in class_correct:
                class_correct[label] += corrected[label_mask].sum().item()
                class_total[label] += label_mask.float().sum().item()
            else:
                class_correct[label] = corrected[label_mask].sum().item()
                class_total[label] = label_mask.float().sum().item()
    
    # Calculate class-wise accuracy
    class_accuracy = {}
    for label in class_total.keys():
        class_accuracy[label] = class_correct[label] / class_total[label]
    
    class_accuracy = {k: v for k, v in sorted(class_accuracy.items(), key=lambda item: item[0])}

    # Calculate micro and macro averages
    for c, acc in class_accuracy.items():
        print(f"Class {c} accuracy: {acc*100:.2f}")

    print(f"Overall: {avg_correct/avg_total:.4f}")
    for g in range(4):
        print(f"Group {g}: {group_correct[g]/group_total[g]:.4f}")
    

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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4) # 2e-3
    parser.add_argument("--ckpt_path", type=str, required=True)
    
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, default="ckpt")
    parser.add_argument("--save_dir", type=str, default="results/class_acc")
    
    parser.add_argument("--neurons", nargs="+", type=int, required=True)
    # parser.add_argument("--neurons_c1", nargs="+", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda3", type=float, default=0.01)
    parser.add_argument("--method", type=str, default="ours", help="ablation studies for focal loss, (cw, focal, ours)")
   
    return parser.parse_args()

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


def train_model(train_loader, val_loader, model, optimizer, ckpt_dir, lambda3, \
                gamma, num_epochs, device, scheduler=None, \
                neuron_mask=None, method="ours"):
    
    num_classes = model.fc.weight.data.shape[0]
    print("num_classes: ", num_classes)
    
    if method=="cw":
        class_weights = torch.ones(num_classes) * 0.95
        class_weights[1] = 1.0
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    elif method=="focal":
        class_weights = [0.95 for _ in range(num_classes)]
        class_weights[1] = 1.0
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    max_iter = len(train_loader)*num_epochs
    best_val_acc = 0.0
    best_cls_acc = 0.0
    ckpt_path = None
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()

        tqdm_loader = tqdm(enumerate(train_loader))
        for t, batch in tqdm_loader:
            optimizer.zero_grad()
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            features = model.forward_features(inputs)
            features = model.global_pool(features)
            outputs = model.fc(features)
            features_masked = features*neuron_mask.detach()
            # features_masked1 = features*neuron_masks[1].detach()
            
            ce_loss = criterion(outputs, labels)
            
            if method=="ours":    
                prob = torch.softmax(outputs, dim=1)
                changed_prob = torch.softmax(model.fc(features_masked), dim=1)
                retaining_ratio = (prob / (changed_prob+1e-9)).mean()
                l2_loss = torch.norm(retaining_ratio-gamma, dim=0, p=2).mean()
                
                loss = ce_loss + l2_loss * lambda3
                tqdm_loader.set_postfix(ce=ce_loss.item(), l1=l2_loss.item())
            else:
                loss = ce_loss
                tqdm_loader.set_postfix(ce=ce_loss.item())
                
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler(t, max_iter)

        model.eval()

        print('Evaluating on validation set...')
        val_acc, class_acc = eval_loop(val_loader, model, device)
        min_class_acc = min(class_acc.values())
        print("Val acc", val_acc*100)
        print(f"Val class acc", class_acc[0]*100, class_acc[1]*100)
        
        if epoch==0:
            continue
        
        # if val_acc >= best_val_acc:
        #     best_val_acc = val_acc
        if best_cls_acc <= min_class_acc:
            best_cls_acc = min_class_acc
            print("Save checkpoint:", val_acc, min_class_acc)
            ckpt_path = os.path.join(ckpt_dir, "best_%.2f.pt" % (val_acc*100))
            torch.save(model, ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
            
    return ckpt_path

def prepare_editing(model, args, viz_paths=None):
    train_dataset, valid_dataset, _ = load_datasets(args.dataset, args.data_dir, args.download_data, g_aug=True)
    if viz_paths is not None:
        train_dataset._image_files.extend(viz_paths)
        train_dataset._labels.extend([args.class_idx for _ in range(len(viz_paths))])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model.eval()
    # Change the classifier layer
    for p in model.fc.parameters():
        p.requires_grad = True
    
    model = model.to(args.device)
    
    embed_dim= model.fc.weight.data.shape[1]
    index = torch.LongTensor(args.neurons)
    neuron_mask = torch.ones(embed_dim)
    neuron_mask.index_fill_(0, index, 0)
    neuron_mask = neuron_mask.unsqueeze(0).to(args.device)
    
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = None#get_warmup_lr_scheduler(optimizer, args.lr)
    
    return train_loader, valid_loader, optimizer, scheduler, neuron_mask


def main(args):
    # set seed
    set_seed(args.seed)

    model = torch.load(args.ckpt_path, map_location=f"cuda:{args.device}")
    model.eval()

    ckpt_dir = os.path.join(args.ckpt_dir, args.dataset+"_"+args.model+"_edit_"+args.method)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # prepare training
    train_loader, valid_loader, optimizer, scheduler, neuron_mask = prepare_editing(model, args)
    
    # Train the model
    new_ckpt_path = train_model(train_loader, valid_loader, model, optimizer, ckpt_dir, \
                                args.lambda3, args.gamma, args.num_epochs,\
                                args.device, scheduler=scheduler, \
                                neuron_mask=neuron_mask, method=args.method)

    test_dataset, _ = load_test_data(args.dataset, args.data_dir, args.download_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    print("Load best checkpoint -", new_ckpt_path)
    model = torch.load(new_ckpt_path, map_location=f"cuda:{args.device}")
    model.eval()
    print('Evaluating on test set...')
    eval_loop_group(test_loader, model, args.device)
    
    
if __name__ == "__main__":
    args = config()
    main(args)