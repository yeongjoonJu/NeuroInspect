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
import random

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from concept_utils import conceptual_counterfactual, ConceptBank


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="bear-bird-cat-dog-elephant:dog(water)")
    parser.add_argument('--out_dir', type=str, default='cce_results')
    
    parser.add_argument('--model-path', type=str, default=None, 
                        help="If provided, will use this model instead of the one in the out_dir")
    
    parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--num-mistakes", default=50, type=int, help="Number of mistakes to evaluate")
    
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    
    ## CCE parameters
    parser.add_argument("--step-size", default=1e-2, type=int, help="Stepsize for CCE")
    parser.add_argument("--alpha", default=1e-1, type=float, help="Alpha parameter for CCE")
    parser.add_argument("--beta", default=1e-2, type=float, help="Beta parameter for CCE")
    parser.add_argument("--n_steps", default=100, type=int, help="Number of steps for CCE")
    parser.add_argument("--edited", action="store_true")

    return parser.parse_args()


    
def main(args):
    sns.set_context("poster")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    model = timm.create_model("resnet18", pretrained=False)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    if args.model_path is None:
        model_path = os.path.join(os.path.join("ckpts", args.dataset), "edited_con.pt" if args.edited else "confounded-model_best.pt") #
    else:
        model_path = args.model_path

    # Load the model
    print(model_path)
    model = torch.load(model_path, map_location=f"cuda:{args.device}")
    model = model.to(args.device)
    model = model.eval()
    
    # Split the model into the backbone and the predictor layer
    
    # Load the concept bank
    concept_bank = ConceptBank(pickle.load(open(args.concept_bank, "rb")), device=args.device)
    
    # Get the images that do not contain the spurious concept
    manager = MetashiftManager()
    train_domain = args.dataset.split(":")[1]
    shift_class = train_domain.split("(")[0]
    spurious_concept = train_domain.split("(")[1][:-1].lower()
    print(f"Shift Class: {shift_class}, Spurious Concept: {spurious_concept}")

    shift_class_images = manager.get_single_class(shift_class, exclude=train_domain)
    np.random.shuffle(shift_class_images)
    
    cls_idx = args.dataset.split(":")[0].split("-").index(shift_class)

    fout = open(f"{args.dataset}_{'edit' if args.edited else 'ori'}_wrong_cases.txt", "wt")
    
    num_eval = 0
    num_correct = 0
    all_ranks = []
    all_scores = []
    tqdm_iterator = tqdm(enumerate(shift_class_images), total=len(shift_class_images))
    for i, image_path in tqdm_iterator:
        # Read the image and label
        image = Image.open(image_path).convert("RGB")
        
        with torch.no_grad():
            image_tensor = transform(image).unsqueeze(0).to(args.device)
            label = cls_idx*torch.ones(1, dtype=torch.long).to(args.device)
            
            # Get the embedding for the image
            embedding = model.forward_features(image_tensor)
            embedding = model.global_pool(embedding)
            # Get the model prediction
            pred = model.fc(embedding).argmax(dim=1)
            
            # Only evaluate over mistakes
            if pred.item() == label.item():
                num_correct += 1
                continue
            
            num_eval += 1
        
        # Run CCE
        explanation = conceptual_counterfactual(embedding, label, concept_bank, model.fc,
                                                alpha=args.alpha, beta=args.beta, step_size=args.step_size,
                                                n_steps=args.n_steps)
        
        # Find the rank of the spurious concept
        all_ranks.append(explanation.concept_scores_list.index(spurious_concept))
        all_scores.append(explanation.concept_scores[spurious_concept])

        if explanation.concept_scores_list.index(spurious_concept) < 3:
            fout.write(f"{i}\n")
        
        tqdm_iterator.set_postfix(mean_rank=1+np.mean(all_ranks), precision_at_3=np.mean(np.array(all_ranks) < 3))
        # if num_eval == args.num_mistakes:
        #     break
    
    fout.close()
    all_ranks = np.array(all_ranks)
    all_scores = np.array(all_scores)
    print(num_correct)
    print(len(shift_class_images))
    print("Acc", num_correct / len(shift_class_images))
    print(f"...:::FINAL => Mean Rank: {1+np.mean(all_ranks):.3f}, Precision@1: {np.mean(all_ranks < 1):.3f}, Precision@3: {np.mean(all_ranks < 3):.3f}, Score: {np.mean(all_scores):.3f}")
    
if __name__ == "__main__":
    args = config()
    main(args)