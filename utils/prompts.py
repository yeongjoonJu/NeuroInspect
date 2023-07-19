import torch

def preprocess_celeba_text(class_name_dict):
    young_class = 39
    gender_class = 20
    eyebrows_classes = [1,12]
    wearing_classes = [15,34,35,36,37,38]
    adjective_classes = [2, 4, 10, 13, 31]
    hair_classes = [5,8,9,11,17,32,33]
    etc_classes = [0,3,6,7,14,16,18,19,21,22,23,25,26,27,28,29,30]
    
    # 24: beard

    if gender_class in class_name_dict:
        if young_class in class_name_dict:
            text = "boy"
        else:
            text = "male"
    else:
        if young_class in class_name_dict:
            text = "girl"
        else:
            text = "female"
    
    adjective_texts = []
    for idx in adjective_classes:
        if idx in class_name_dict:
            adjective_texts.append(class_name_dict[idx])
    if adjective_texts:
        text = ", ".join(adjective_texts)+ " " + text

    eyebrows_texts = []
    for idx in eyebrows_classes:
        if idx in class_name_dict:
            eyebrows_classes.append(class_name_dict[idx])
    if eyebrows_texts:
        eyebrows_texts = ", ".join(eyebrows_texts) + " eyebrows"
    
    hair_texts = []
    for idx in hair_classes:
        if idx in class_name_dict:
            hair_classes.append(class_name_dict[idx])
    if hair_texts:
        hair_texts = ", ".join(hair_texts) + " hair"
    
    wearing_texts = []
    for idx in wearing_classes:
        if idx in class_name_dict:
            wearing_classes.append(class_name_dict[idx])
    if wearing_texts:
        wearing_texts = "wearing " + ", ".join(wearing_texts)
    
    with_texts = []
    for idx in etc_classes:
        if idx in class_name_dict:
            with_texts.append(class_name_dict[idx])
    
    if 24 in class_name_dict:
        with_texts.append("beard")
        
    if with_texts:
        with_texts = [", ".join(with_texts)]
    if eyebrows_texts:
        with_texts.append(eyebrows_texts)
    if hair_texts:
        with_texts.append(hair_texts)
    
    if with_texts:
        text += " with " + ", and ".join(with_texts)
    
    if wearing_texts:
        text += ", " + wearing_texts
    
    return text


def prepare_multi_label_task(logits, class_dict):
    probs = torch.sigmoid(logits)
    selected = probs>0.5
    class_name_dict = {}
    num_classes = len(class_dict.keys())
    target_labels = torch.zeros(num_classes)
    class_indices = torch.arange(0,num_classes).to(selected.device)
    class_selected = class_indices[selected]

    for class_idx in class_selected.tolist():
        target_labels[class_idx] = 1.0
        class_name_dict[class_idx] = class_dict[class_idx]
    if not class_name_dict:
        class_idx = torch.argmax(logits,dim=1)[0].item()
        target_labels[class_idx] = 1.0
        class_name_dict[class_idx] = class_dict[class_idx]

    class_name = preprocess_celeba_text(class_name_dict)
    
    return class_name, target_labels


def prepare_single_label_task(logits, class_dict):
    target_label = torch.argmax(logits, dim=0).item()
    class_name = class_dict[target_label]
    
    return class_name, target_label


def prepare_class_names(class_indices, class_dict):
    class_names = []
    for class_idx in class_indices:
        class_name = class_dict[class_idx.item()]
        class_names.append(class_name)
        
    return class_names
