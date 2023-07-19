import os, argparse
import warnings

from utils.model_utils import load_vision_model
warnings.filterwarnings("ignore", category=UserWarning)

from clip_illusion import Illusion

from utils.objectives import ClassConditionalObjective

import torch
import numpy as np

from utils.config import Domain2Dict

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--iters", type=int, default=2048)
    parser.add_argument("--dream_iters", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--exp_name", type=str, default="dis")
    parser.add_argument("--probing_layer", type=str, default="layer4")
    parser.add_argument("--from_class", type=int, default=None)
    parser.add_argument("--to_class", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--domain", type=str, default="imagenet")
    parser.add_argument("--target_class", type=int, default=None)
    parser.add_argument("--class_gamma", type=float, default=0.8)
    parser.add_argument("--target_neurons", nargs="+", type=int, default=None)
    parser.add_argument("--domain_eps", type=float, default=0.01)
    args = parser.parse_args()
    
    assert args.target_neurons is not None

    torch.manual_seed(args.seed)

    exp_name = f"{args.exp_name}_{args.model}.{args.probing_layer}"

    device_id=f"cuda:{args.device}"
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    model_name = args.model

    embed_dim = 512
    if "resnet50" in model_name:
        embed_dim = 2048
    elif model_name == "efficientnet_b3a":
        embed_dim = 1536

    class_dict = Domain2Dict[args.domain]

    # load model
    transform, model, pool, decision, config = load_vision_model(model_name, device=device, ckpt_path=args.ckpt_path)
    illusion = Illusion(model, decision, \
                        ClassConditionalObjective(image_size=224, is_vit=False, class_dict=class_dict, \
                        class_gamma=args.class_gamma, domain_eps=args.domain_eps,), \
                        device=device, img_mean=config["mean"], img_std=config["std"])
    """
    Acquiring data-agnostic neuron representations
    """
    if args.target_neurons[0]==-1:
        args.target_neurons = list(range(embed_dim))
        
    # Class-oriented neuron representations
    illusion.visualize_neurons(
        experiment_name=exp_name,
        target_neurons=args.target_neurons,
        layer= eval(f"model.{args.probing_layer}"), # model.layer4,
        iters=args.iters, # 80,
        lr=args.lr, #lr=5e-3,
        overwrite_experiment=True,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay, # 3e-2
        quiet=args.quiet,
        class_idx=args.target_class
    )