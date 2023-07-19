import argparse
import numpy as np
from PIL import Image
import torch, json
import torch.nn.functional as F
import sys
sys.path.append("exp_spurious")
import random
from dataset import MetashiftManager
from utils.model_utils import load_vision_model
from utils.feature_manipulation import *

def inspect_wrong_cases(shift_class_images, transform, model, pool, decision, device):
    correct_top1 = 0
    total = 0

    num_images = len(shift_class_images)
    class_logits = []
    wrong_cases = []
    with torch.no_grad():
        for idx, image_path in enumerate(shift_class_images):
            print("\r%d / %d" % (idx+1, num_images), end="")
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0)
            image = image.to(device)

            features = model.forward_features(image)
            features = pool(features)
            outputs = decision(features)
            
            # rank 1
            _, pred = torch.max(outputs, dim=1)
            labels = torch.LongTensor([cls_idx]).to(device)

            total += labels.size(0)
            correct_top1 += (pred == labels).sum().item()

            if (pred!=labels)[0].item():
                wrong_cases.append(shift_class_images[idx])
                class_logits.append((features[0]*decision.weight.data[cls_idx]).unsqueeze(0))
            
    print("\ntop-1 percentage :  {0:0.2f}%".format(correct_top1 / total * 100))
    class_logits = torch.cat(class_logits)

    return wrong_cases, class_logits


def vote_topk(adjust_w, neg=False, k=5):
    vote = {"top-1":{}, "top-2": {}, "top-3": {}, "top-4": {}, "top-5": {}}
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

def aggregate_vote(vote, neurons):
    total_topk = {}
    for key in ["top-1", "top-2", "top-3", "top-4", "top-5"]:
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
    
    return total_topk
                

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--dataset", type=str, default="bear-bird-cat-dog-elephant:dog(snow)")
    parser.add_argument("--save_name", type=str, default="debugging")
    parser.add_argument("--ckpt_name", type=str, default=None)
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
    if args.ckpt_name is None:
        model_path = f"exp_spurious/ckpts/{args.dataset}/confounded-model_best.pt"
    else:
        model_path = f"exp_spurious/ckpts/{args.dataset}/{args.ckpt_name}"

    transform, model, pool, decision, config = load_vision_model(args.model, device, ckpt_path=model_path)
    num_classes, embed_dim = decision.weight.data.shape

    # Get the images that do not contain the spurious concept
    manager = MetashiftManager()
    classes, train_domain = args.dataset.split(":")
    classes = classes.split("-")
    num_classes = len(classes)
    shift_class = train_domain.split("(")[0]
    spurious_concept = train_domain.split("(")[1][:-1].lower()
    print(f"Shift Class: {shift_class}, Spurious Concept: {spurious_concept}")

    shift_class_images = manager.get_single_class(shift_class, exclude=train_domain)
    # np.random.shuffle(shift_class_images)
    cls_idx = args.dataset.split(":")[0].split("-").index(shift_class)

    # Wrong case inspection
    wrong_cases, class_logits = inspect_wrong_cases(shift_class_images, \
                                    transform, model, pool, decision, device)

    num_total = len(shift_class_images)
    print("Accuracy:", (num_total-len(wrong_cases))/num_total)
    print("The number of wrong cases:", len(wrong_cases))
    
    # Counterfactual explanation
    adjust_w = counterfactual_explanation(wrong_cases, transform, model, pool, \
                                        decision, device, class_idx=cls_idx)

    pos_vote = vote_topk(adjust_w, neg=False)
    neg_vote = vote_topk(adjust_w, neg=True)

    total_topk_pos = aggregate_vote(pos_vote, adjust_w.shape[0])
    total_topk_neg = aggregate_vote(neg_vote, adjust_w.shape[0])

    report = {"total": {"+":total_topk_pos, "-":total_topk_neg}, "details": {"+":pos_vote, "-":neg_vote}}
    
    save_path = "/".join(model_path.split("/")[:-1])
    name = args.save_name if args.class_attr_path is None else f"{args.save_name}_attr"
    save_path = f"{save_path}/{name}.json"
    with open(save_path, "w") as fout:
        json.dump(report, fout, indent=2)