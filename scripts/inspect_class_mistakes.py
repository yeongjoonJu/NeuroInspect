import argparse, os
import numpy as np
from PIL import Image
import torch, json
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys
import random
from utils.model_utils import load_vision_model
from utils.feature_manipulation import counterfactual_explanation
from train_test_classifier import load_test_data, eval_loop

def get_feature_importance(features, decision, class_idx, embed_dim, device):
    logits = decision(features)
    probs = torch.softmax(logits, dim=1)[:,class_idx] # original prob
    
    features = features.unsqueeze(1)
    features = features.expand(-1, embed_dim, -1).cpu() # b d d
    mask = torch.eye(embed_dim, dtype=torch.bool)
    mask = mask.unsqueeze(0)
    mask = mask.expand(features.size(0), -1, -1)
    features[mask] = 0.0
    features = features.view(-1, embed_dim)
    logits = decision(features.to(device))
    changed_prob = torch.softmax(logits, dim=1)[:, class_idx] # b*embed_dim
    changed_prob = changed_prob.view(-1, embed_dim)
    probs = probs.unsqueeze(-1)
    retaining_ratio = (probs / changed_prob).detach().cpu()

    return retaining_ratio[0]

def inspect_mistakes_in_class(class_idx, num_classes, loader, img_filelist, model, device, decision=None):
    model.eval()
    tqdm_loader = tqdm(enumerate(loader), desc="Evaluating")
    
    # Initialize dictionary for storing class-wise accuracy and count
    mistakes = []
    mislabels = []
    agg_features = None
    n_samples = 0
    class_features = {}
    
    with torch.no_grad():
        for idx, (inputs, labels) in tqdm_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            features = model.forward_features(inputs)
            outputs = model.forward_head(features)

            preds = torch.argmax(outputs, 1)
            
            end_idx = (idx+1)*loader.batch_size
            if end_idx > len(loader.dataset):
                end_idx = len(loader.dataset)
            indices = torch.arange(idx*loader.batch_size, end_idx)
            incorrect = torch.logical_and(preds!=labels, labels==class_idx).cpu()
            mistakes.extend(indices[incorrect].tolist())
            mislabels.extend(preds[incorrect].tolist())

            if len(features.shape) == 4:
                B,C,_,_ = features.shape
                fl_features = features.view(B, C, -1).mean(-1)
            else:
                fl_features = features[:,0]
                B, C = fl_features.shape
                
            n_samples += B
            if agg_features is None:
                agg_features = fl_features.sum(0).detach()
            else:
                agg_features = agg_features + fl_features.sum(0).detach()
            
            indices = torch.arange(labels.shape[0])
            for c in range(num_classes):
                c_feats = fl_features[indices[labels==c]].detach()
                if c in class_features:
                    class_features[c]["count"] += c_feats.shape[0]
                    class_features[c]["features"] += c_feats.sum(0,keepdim=True).detach()
                else:
                    class_features[c] = {}
                    class_features[c]["count"] = c_feats.shape[0]
                    class_features[c]["features"] = c_feats.sum(0,keepdim=True).detach()
        
    agg_features = agg_features / n_samples
    
    mistake_samples = []
    for idx in mistakes:
        mistake_samples.append(img_filelist[idx][0] if type(img_filelist[idx]) is tuple else img_filelist[idx])
        
    if decision is None:
        decision = model.fc
    
    embed_dim = decision.weight.shape[1]
    imp = get_feature_importance(agg_features.unsqueeze(0), decision, class_idx, embed_dim, device)
    
    # Get class-wise features
    class_representatives = []
    for c in range(num_classes):
        class_representatives.append(class_features[c]["features"] / class_features[c]["count"])
    
    return mistake_samples, mislabels, imp, class_representatives
    

def vote_topk(adjust_w, neg=False, k=5):
    vote = {}
    for tk in range(k):
        vote[f"top-{tk+1}"] = {}
        
    for i in range(adjust_w.shape[0]):
        _, n_idx = torch.topk(adjust_w[i], k=k, dim=0, sorted=True,
                              largest=False if neg else True)
        for r in range(k):
            key = n_idx[r].item()
            if neg and adjust_w[i][key].item() > 0:
                continue
            if not key in vote[f"top-{r+1}"]:
                vote[f"top-{r+1}"][key] = 1
            else:
                vote[f"top-{r+1}"][key] += 1
                
    return vote

def aggregate_vote(vote, neurons, to_text=False):
    total_topk = {}
    for key in vote.keys():
        for n_id, cnt in vote[key].items():
            if n_id in total_topk:
                total_topk[n_id] += cnt
            else:
                total_topk[n_id] = cnt

    except_neurons = []
    for n_id in total_topk.keys():
        ratio = total_topk[n_id]/neurons*100
        if ratio > 3.0:
            total_topk[n_id] = ratio
        else:
            except_neurons.append(n_id)
            
    for n_id in except_neurons:
        del total_topk[n_id]
        
    aligned = sorted(total_topk.items(), key=lambda x: x[1], reverse=True)
    if to_text:
        total_topk = {k:"%.3f%%" % v for k,v in aligned}
    else:
        total_topk = {k:v for k,v in aligned}
    
    return total_topk
                

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="results/class_imbalance")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--class_idx", type=int, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load model
    transform, model, pool, decision, config = load_vision_model(args.model, device, ckpt_path=args.ckpt_path)
    num_classes, embed_dim = decision.weight.data.shape

    # load dataset
    test_dataset, classes = load_test_data(args.dataset, args.data_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    # if args.domain=="imagenet":
    #     img_filelist = test_dataset.imgs
    # else:
    img_filelist = test_dataset._image_files
    
    # Test the model
    mistake_samples, _ = inspect_mistakes_in_class(args.class_idx, test_loader, img_filelist, model, device)
    num_mistakes = len(mistake_samples)
    print(f"{args.model} has {num_mistakes} mistakes for the {args.dataset} dataset.")

    # Counterfactual explanation
    adjust_w = counterfactual_explanation(mistake_samples, transform, model, pool, \
                                        decision, device, class_idx=args.class_idx)

    pos_vote = vote_topk(adjust_w, neg=False)
    neg_vote = vote_topk(adjust_w, neg=True)

    total_topk_pos = aggregate_vote(pos_vote, adjust_w.shape[0], to_text=True)
    total_topk_neg = aggregate_vote(neg_vote, adjust_w.shape[0], to_text=True)

    report = {"total": {"+":total_topk_pos, "-":total_topk_neg}, "details": {"+":pos_vote, "-":neg_vote}}
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = f"{args.save_dir}/{args.dataset}_{args.model}_c{args.class_idx}.json"
    with open(save_path, "w") as fout:
        json.dump(report, fout, indent=2)