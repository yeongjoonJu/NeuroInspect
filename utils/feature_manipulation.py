from PIL import Image
import torch
import torch.nn.functional as F


def compute_contribution(features, decision):
    return decision.weight*features + decision.bias.unsqueeze(-1)

def compute_elastic_loss(w, logits, label, alpha=1e-1, beta=1e-2):
    ce_loss = F.cross_entropy(logits, label)
    
    # elastic net regularization
    if w.shape[1] > 1:
        l1_loss = torch.norm(w, dim=1, p=1)/w.shape[1]
        l2_loss = torch.norm(w, dim=1, p=2)/w.shape[1]
        loss = ce_loss + l1_loss.sum()*alpha + l2_loss.sum()*beta
    else:
        l1_loss = torch.norm(w, dim=1, p=1)
        l2_loss = torch.norm(w, dim=1, p=2)
        loss = ce_loss + l1_loss.mean()*alpha + l2_loss.mean()*beta
        
    return loss

def counterfactual_explanation(wrong_cases, transform, model, pool, decision, device, class_idx=None, correct_labels=None, lr=0.01, momentum=0.9):
    assert class_idx is not None or correct_labels is not None
    
    label_idx = class_idx
    
    total = len(wrong_cases)
    adjust_w = []
    
    for i, image_path in enumerate(wrong_cases):
        image_pil = Image.open(image_path).convert("RGB")
        image = transform(image_pil).unsqueeze(0)
        image = image.to(device)

        if correct_labels is not None:
            label_idx = correct_labels[i]
            
        label = torch.LongTensor([label_idx]).to(device)

        # num_classes = model.fc.weight.data.shape[0]
        w = torch.zeros(1, model.num_features).to(device)
        w = torch.nn.Parameter(w, requires_grad=True)
        optimizer = torch.optim.SGD([w], lr=lr, momentum=momentum)

        with torch.no_grad():
            features = model.forward_features(image)
            features = pool(features)
        
        """
        contrib = features * decision.weight.data[class_idx:class_idx+1] # contribution
        if class_attr is not None:
            contrib = contrib*class_attr[class_idx:class_idx+1]
        c_max = contrib[0].max().item() # value
        """

        optim_iter = 0
        while True:
            optimizer.zero_grad()
            features = model.forward_features(image)
            features = pool(features)
            revised_features = features + w
            logits = decision(revised_features)

            pred = torch.argmax(logits, dim=1)
            if pred[0].item()==label_idx:
                break

            loss = compute_elastic_loss(w, logits, label)
            loss.backward()
            optimizer.step()

            prev_w = w.data
            w.data = prev_w.detach()
            w.grad.zero_()
            
            optim_iter += 1

        if optim_iter!=0:
            adjust_w.append(w.data.detach())
        
        print("\rProgress %.2f Optimizing iteration: %d, top neuron: %d   " % (((i+1)/total)*100, optim_iter, w.data.max(dim=1)[1][0].item()), end="")
    print()
    adjust_w = torch.cat(adjust_w).cpu() # N F
    
    return adjust_w