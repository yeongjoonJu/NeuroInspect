import argparse, random, os
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms

from clip_illusion import Illusion
from utils.model_utils import load_vision_model
from utils.objectives import ClassConditionalObjective
from utils.config import Domain2Dict
from utils.feature_manipulation import counterfactual_explanation
from train_test_classifier import load_test_data, eval_loop
from scripts.inspect_class_mistakes import inspect_mistakes_in_class, vote_topk, aggregate_vote
from scripts.edit_decision import prepare_editing, train_model


def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--class_idx", type=int, required=True)
    parser.add_argument("--class_gamma", type=float, default=0.5)
    parser.add_argument("--domain_eps", type=float, default=0.05)
    
    # Model and data details
    md_parser = parser.add_argument_group("Model and data details")
    md_parser.add_argument("--dataset", type=str, required=True, help="dataset name")
    md_parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    md_parser.add_argument("--model", type=str, default="resnet50", help="model name")
    md_parser.add_argument("--data_dir", type=str, default="data", help="data directory")
    md_parser.add_argument("--test_batch_size", type=int, default=128, help="test batch size")
    md_parser.add_argument("--lr", type=float, default=1e-4, help="editing lr")
    
    # CLIP-Illusion hyperparameters
    ci_parser = parser.add_argument_group("CLIP-Illusion hyperparameters")
    ci_parser.add_argument("--iters", type=int, default=450, help="number of iterations")
    ci_parser.add_argument("--domain", type=str, default="imagenet", help="domain of the dataset")
    ci_parser.add_argument("--save_dir", type=str, default="results", help="directory to save results")
    ci_parser.add_argument("--batch_size", type=int, default=3)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    return args

def prepare_model_and_illusion(args):
    device = torch.device(args.device)
    class_dict = Domain2Dict[args.domain]
    transform, model, pool, decision, config = load_vision_model(args.model, device=device, ckpt_path=args.ckpt_path)
    illusion = Illusion(model, decision, ClassConditionalObjective(image_size=224, domain_eps=args.domain_eps), \
        device=args.device, class_dict=class_dict, img_mean=config["mean"], img_std=config["std"])
    
    return model, illusion, {"transform": transform, "pool": pool, "decision": decision}

def inference_one_sample(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0).to(device)
    
    logits = model(image)
    pred = torch.argmax(logits, dim=-1)[0].item()
    
    return pred


def select_spurious_attribute(model, illusion, images, acts, masks, cor_feats, inc_feats, batch_size, reduction="mean"):
    with torch.no_grad():
        mean_class_vector = (cor_feats.sum(dim=0) + inc_feats.sum(dim=0)) / (cor_feats.shape[0] + inc_feats.shape[0])
        mean_class_vector = mean_class_vector.unsqueeze(0)
        features     = model.forward_features(illusion.postprocess(images))
        features_rev = model.forward_features(illusion.postprocess(images*masks))
        
        B, C, _, _ = features.shape
        if batch_size==1:
            features = features.view(B, C, -1).mean(dim=-1)
            features_rev = features_rev.view(B, C, -1).mean(dim=-1)
            acts = acts.squeeze(-1)
        else:
            num_n = B//batch_size
            acts = acts.view(num_n, batch_size)
            features = features.view(num_n, batch_size, C, -1).mean(dim=-1)
            features_rev = features_rev.view(num_n, batch_size, C, -1).mean(dim=-1)
            if reduction=="max":
                acts, index = acts.max(dim=1)
                index = index.view(num_n,1,1).expand(-1, 1, C)
                features = torch.gather(features, dim=1, index=index).squeeze(1)
                features_rev = torch.gather(features_rev, dim=1, index=index).squeeze(1)
            else:
                acts, _ = acts.min(dim=1)
                features = features.mean(dim=1)
                features_rev = features_rev.mean(dim=1)
            
        corr = F.cosine_similarity(features, mean_class_vector.detach())
        corr_rev = F.cosine_similarity(features_rev, mean_class_vector.detach())
        ratio = corr_rev/(corr+1e-9)
        print(corr)
        print(corr_rev)
        print(ratio)
        
        selected = torch.logical_or(ratio < 0.95, acts==0.0)
        if selected.sum()==0:
            _, index = torch.topk(ratio, dim=0, largest=False, k=3)
            selected[index] = True
                        
        return selected

if __name__=="__main__":
    args = parsing_args()
    
    # Prepare model and CLIP-Illusion
    model, illusion, modules = prepare_model_and_illusion(args)
    dimension = modules["decision"].weight.data.shape[1]
    
    # Prepare test dataset
    test_dataset, classes = load_test_data(args.dataset, args.data_dir, split="valid")
    print("> The number of test samples: %d" % len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    
    img_filelist = test_dataset._image_files
    
    # set device
    device = torch.device(args.device)
    
    # Test the model
    mistake_samples, imp, inc_feats, cor_feats = inspect_mistakes_in_class(args.class_idx, test_loader, img_filelist, model, device)
    num_mistakes = len(mistake_samples)
    print(f"> {args.model} has {num_mistakes} mistakes for the {args.dataset} dataset.")
    
    class_name = illusion.class_dict[args.class_idx].split(", ")[-1]
    
    # Counterfactual explanation
    print(f"> Inspecting mistakes for {class_name}...")
    adjust_w = counterfactual_explanation(mistake_samples, modules["transform"], model, modules["pool"], \
                                        modules["decision"], device, class_idx=args.class_idx)
    pos_vote = vote_topk(adjust_w, neg=False, k=5)
    neg_vote = vote_topk(adjust_w, neg=True, k=5)
    total_topk_pos = aggregate_vote(pos_vote, adjust_w.shape[0])
    total_topk_neg = aggregate_vote(neg_vote, adjust_w.shape[0])
    pos_ids = total_topk_pos.keys()
    neg_ids = total_topk_neg.keys()
    
    folder_class_name = class_name.replace(" ", "_")
    exp_name = f"debug_{args.model}_{args.dataset}_{folder_class_name}"
    
    print("\n<Underlying Reasons>")
        
    print("\n- Insufficient Properties:")
    print(" Rank    ID  Prec@5 %  Importance")
    print("--------------------------------")
    skip = 0
    pos_n_id = []
    for r, (n_id, v) in enumerate(total_topk_pos.items()):
        if n_id in neg_ids:
            skip+=1
            continue
        if r >= 15:
            break
        pos_n_id.append(n_id)
        print("Top-%d: %4d  %2.3f   %.4f" % (r+1-skip, n_id, v, imp[n_id]))
    print("--------------------------------")
            
    print("\n- Excessive Properties:")
    print(" Rank    ID  Prec@5 %  Importance")
    print("--------------------------------")
    skip = 0
    for r, (n_id, v) in enumerate(total_topk_neg.items()):
        if n_id in pos_ids:
            skip+=1
            continue
        if r >= 15:
            break
        print("Top-%d: %4d  %2.3f   %.4f" % (r+1-skip, n_id, v, imp[n_id]))
    print("--------------------------------\n")
        
    print("> Debugging mode.")
    print("> Please enter exit to quit.\n")
    cmd = ""
    while True:
        cmd = input(">> Input indices of neuron to debug: ")
        try:
            if cmd == "exit":
                print("> Bye.")
                break
            if cmd == "auto":
                print(pos_n_id[:10])
                neuron_ids = pos_n_id[:10]
            else:
                neuron_ids = cmd.split(" ")
                neuron_ids = [int(n_id) for n_id in neuron_ids]
            
            check = True
            for n_id in neuron_ids:
                if n_id >= dimension:
                    print("> Please input an integer less than %d or exit." % dimension)
                    check=False
                    break
            if not check:
                continue
        except ValueError as e:
            print("> Please input an integer or exit.")
            continue
        
        cmd = input(">> Do you want to condition the answer class? (y/n): ")
        if cmd=="n":
            cond_class = None
        else:
            cond_class = args.class_idx
        
        # CLIP-Illusion
        print("------------------")
        print(f"\n> Please look at the directory {exp_name}.\n")
        viz, acts, masks = illusion.visualize_neurons(
            experiment_name=exp_name,
            target_neurons=neuron_ids,
            layer=model.layer4,
            iters=args.iters,
            lr=9e-3,
            batch_size=args.batch_size,
            weight_decay=2e-4,
            class_idx=cond_class,
            thresholding=False,
            quiet=True,
        )
        print("> Done.\n")
        
        # Automatic Neuron selection
        cmd = input(">> Do you want to select neurons to edit automatically? (y/n): ")
        if cmd.strip()=="y":
            print(f"\n> Selecting spurious attributes...")
            n_mask = select_spurious_attribute(model, illusion, viz, acts, masks, cor_feats, inc_feats, args.batch_size)
            args.neurons = torch.LongTensor(neuron_ids)[n_mask.cpu()].tolist()
            
            print(f"\n> Preparing training... | Selected neurons: {args.neurons}\n")
            if args.ckpt_path is None:
                args.ckpt_path = f"./ckpt/{args.dataset}"
                os.makedirs(args.ckpt_path, exist_ok=True)

            args.lr = 1e-4
            args.download_data = True
            ckpt_dir = os.path.dirname(args.ckpt_path)
            lambda3 = 0.1
            num_epochs = 20
            
            gamma = (min([imp[n_id] for n_id in args.neurons])-1)/3 + 1
            print(gamma)
            
            # Edit decision
            ill_batch_size = args.batch_size
            args.batch_size = args.test_batch_size
            train_loader, valid_loader, optimizer, scheduler, neuron_mask = prepare_editing(model, args)
            new_ckpt_path = train_model(train_loader, valid_loader, model, optimizer, ckpt_dir, lambda3, gamma, num_epochs,\
                                    scheduler=scheduler, device=args.device, mod_class=args.class_idx, neuron_mask=neuron_mask)
            print("Best checkpoint -", new_ckpt_path)
            
            args.batch_size = ill_batch_size