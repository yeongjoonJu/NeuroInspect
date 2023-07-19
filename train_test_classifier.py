import argparse, random, os
from glob import glob
import shutil, json
import numpy as np
from tqdm import tqdm
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import ImageFolder
from torchvision import transforms

from dataset.general import Flowers102, Food101, SUN397, Waterbird
from dataset.confounder import CelebADataset, CUBDataset, DRODataset

from scipy.io import loadmat

confound_dict = {
    'CelebA':{
        'constructor': CelebADataset,
        'target_name': "Blond_Hair",
        "confounder_names": ["Male"]
    },
    'waterbird':{
        'constructor': CUBDataset,
        'target_name': "waterbird_complete95",
        "confounder_names": ["forest2water2"]
    },
}

data_dict = {
    # Oxford Flowers 102 Dataset
    "flowers102": Flowers102,
    # Food-101 Dataset
    "food101": Food101,
    # The Scene UNderstanding (SUN) DataSet
    "SUN397": SUN397,
    # ImageNet validation
    "imagenet": ImageFolder,
    # Lego bricks
    "lego": ImageFolder,
    "waterbird": Waterbird, "celeba": None
}

def prepare_confounder_data(name: str, root: str):
    data_args = confound_dict[name]
    full_dataset = data_args['constructor'](
        root_dir=root,
        target_name=data_args['target_name'],
        confounder_names=data_args['confounder_names'],
        augment_data=False)

    splits = ['train', 'val', 'test']
    subsets = full_dataset.get_splits(splits)
    dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
                              n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
                   for split in splits]
    return dro_subsets


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    
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
        
def download_and_split_SUN397(data_dir):
    os.makedirs(f"{data_dir}/SUN397_split/test/SUN397", exist_ok=True)
    if not os.path.exists(f"{data_dir}/SUN397_split/SUN397"):
        download_and_extract_archive("https://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz", \
                                    f"{data_dir}/SUN397_split", md5="8ca2778205c41d23104230ba66911c7a")
    
    if not os.path.exists(f"{data_dir}/SUN397_split/Partitions.zip"):
        download_and_extract_archive("https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip", \
                                        f"{data_dir}/SUN397_split")
    
    if not os.path.exists(f"{data_dir}/SUN397_split/test/SUN397/ClassName.txt"):
        print("Spliting train and test set for SUN397 dataset...")
        test_filelist = []
        with open(f"{data_dir}/SUN397_split/Testing_01.txt", "r") as f:
            test_filelist = [f"{data_dir}/SUN397_split/SUN397{line.strip()}" for line in f.readlines()]
        
        for test_file in tqdm(test_filelist):
            dir_name = os.path.dirname(test_file).replace("SUN397_split/SUN397", "SUN397_split/test/SUN397")
            if os.path.exists(os.path.join(dir_name, os.path.basename(test_file))):
                continue
            os.makedirs(dir_name, exist_ok=True)
            shutil.move(test_file, dir_name)
            
        shutil.copy(f"{data_dir}/SUN397_split/SUN397/ClassName.txt", f"{data_dir}/SUN397_split/test/SUN397")
    
    

def load_datasets(dataset_name, data_path, download=True, transform=None):
    split_train_str = "train"
    split_valid_str = "valid"
    
    if dataset_name in ["flowers102", "celeba"]:
        split_valid_str = "val"
        
    if transform is not None:
        train_transform = transform
        test_transform = transform
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=15, translate=(0.15,0.15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([ \
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    data_class = data_dict[dataset_name]
    
    if dataset_name=="SUN397":
        download_and_split_SUN397(data_path)
        data_path = f"{data_path}/SUN397_split"
        
    if dataset_name in ["SUN397", "lego"]:
        # data manual split
        dataset = data_class(data_path, transform=train_transform)
        num_data = len(dataset)
        training_ratio = 0.8
        num_train_data = int(num_data*training_ratio)
        num_valid_data = num_data - num_train_data
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [num_train_data, num_valid_data])
        valid_dataset.transform = test_transform
        classes = dataset.classes
    elif dataset_name in ["celeba"]:
        train_dataset, valid_dataset, _ = prepare_confounder_data(dataset_name, data_path)
        classes = ["landbird", "waterbird"]
    elif dataset_name=="food101":
        train_dataset = data_class(data_path, split=split_train_str, download=download, transform=train_transform)
        if train_dataset.export_valid_meta_data():
            train_dataset = data_class(data_path, split=split_train_str, download=download, transform=train_transform)
        valid_dataset = data_class(data_path, split=split_valid_str, download=download, transform=test_transform)
        classes = train_dataset.classes        
    else:
        train_dataset = data_class(data_path, split=split_train_str, download=download, transform=train_transform)
        valid_dataset = data_class(data_path, split=split_valid_str, download=download, transform=test_transform)
        
        if dataset_name=="flowers102":
            labels = loadmat(train_dataset._base_folder / train_dataset._file_dict["label"][0], squeeze_me=True)
            classes = sorted(list(set((labels["labels"] - 1).tolist())))
        else:
            classes = train_dataset.classes
            
    return train_dataset, valid_dataset, classes


def load_test_data(dataset_name, data_path, download=True, transform=None, split="test"):
    if split=="valid" and dataset_name in ["flowers102", "celeba"]:
        split = "val"
        
    if transform is None:
        transform = transforms.Compose([ \
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    classes = None
    data_class = data_dict[dataset_name]
    if dataset_name in ["imagenet", "lego"]:
        test_dataset = ImageFolder(data_path, transform=transform)
    elif dataset_name in ["celeba"]:
        _, valid_dataset, test_dataset = prepare_confounder_data(dataset_name, data_path)
        if split=="valid":
            test_dataset = valid_dataset
        classes = ["landbird", "waterbird"]
    elif dataset_name=="SUN397":
        data_path = f"{data_path}/SUN397_split/test"
        test_dataset = data_class(data_path, download=False, transform=transform)
    else:
        test_dataset = data_class(data_path, split=split, download=download, transform=transform)
        
    if dataset_name=="flowers102":
        labels = loadmat(test_dataset._base_folder / test_dataset._file_dict["label"][0], squeeze_me=True)
        classes = sorted(list(set((labels["labels"] - 1).tolist())))
    elif classes is None:
        classes = test_dataset.classes
        
    return test_dataset, classes


@torch.no_grad()
def eval_loop(loader, model, device, out_mistakes=False):
    model.eval()
    accuracymeter = AverageMeter()
    tqdm_loader = tqdm(enumerate(loader), desc="Evaluating")
    
    # Initialize dictionary for storing class-wise accuracy and count
    class_correct = {}
    class_total = {}
    mistakes = []

    for idx, batch in tqdm_loader:
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        outputs = model(inputs)

        preds = torch.argmax(outputs, 1)
        
        # Update overall accuracy and loss meters
        accuracymeter.update((preds == labels).float().mean().cpu().numpy(), inputs.size(0))
        
        # Update class-wise accuracy and count
        for i in range(len(labels)):
            label = labels[i].item()
            if label in class_correct:
                class_correct[label] += (preds[i] == label).float().sum().item()
                class_total[label] += 1
            else:
                class_correct[label] = (preds[i] == label).float().sum().item()
                class_total[label] = 1
                
        if out_mistakes:
            end_idx = (idx+1)*loader.batch_size
            if end_idx > len(loader.dataset):
                end_idx = len(loader.dataset)
            indices = torch.arange(idx*loader.batch_size, end_idx)
            incorrect = (preds!=labels).cpu()
            mistakes.extend(indices[incorrect].tolist())
        
        tqdm_loader.set_postfix(Acc=accuracymeter.avg)
    
    # Calculate class-wise accuracy
    class_accuracy = {}
    for label in class_total.keys():
        class_accuracy[label] = class_correct[label] / class_total[label]
    
    class_accuracy = {k: v for k, v in sorted(class_accuracy.items(), key=lambda item: item[0])}

    if out_mistakes:
        return mistakes, accuracymeter.avg
    else:
        # Return overall accuracy and class-wise accuracy
        return accuracymeter.avg, class_accuracy


def train_model(train_loader, valid_loader, model, optimizer, scheduler, num_epochs, device, save_dir):
    criterion = torch.nn.CrossEntropyLoss() 

    best_val_acc = 0.0    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()
  
        tqdm_loader = tqdm(train_loader)
        for batch in tqdm_loader:
            optimizer.zero_grad()
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            tqdm_loader.set_postfix(loss=loss.item())

        model.eval()
        print('Evaluating on validation set...')
        val_acc, _ = eval_loop(valid_loader, model, device)
        if val_acc > 0.7 and val_acc > best_val_acc:
            best_val_acc = val_acc

            # Save the model and configurations
            print("Save checkpoint", val_acc)
            torch.save(model, os.path.join(save_dir, "best_model.pt"))
            
def test_model(test_loader, model, device):
    model.eval()
    print('Evaluating on test set...')
    test_acc, class_accuracy = eval_loop(test_loader, model, device)
    print("Test accuracy", test_acc)
    return class_accuracy
            

def change_decision_layer(model, num_classes, requires_grad=True):
    if isinstance(model, nn.DataParallel):
        model.module.fc = nn.Linear(model.module.fc.in_features, num_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    for p in model.fc.parameters():
        p.requires_grad = requires_grad
        
    return model


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-3) # 2e-3
    parser.add_argument("--eta_min_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, default="ckpt")
    parser.add_argument("--save_dir", type=str, default="results/class_acc")
    
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # set seed
    set_seed(args.seed)
    
    # load model
    model = timm.create_model(args.model, pretrained=True)
    
    if args.test_only:
        test_dataset, classes = load_test_data(args.dataset, args.data_dir, args.download_data)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        # # Change the classifier layer
        # change_decision_layer(model, len(classes), requires_grad=True)
        
        if args.ckpt_path is not None:
            model = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
        else:
            raise ValueError("Please specify the checkpoint path")
        
        # set model
        model = model.to(args.device)
        model.eval()
        
        class_acc = test_model(test_loader, model, args.device)
        ckpt_name = "_".join(os.path.basename(args.ckpt_path).split(".")[:-1])
        with open(f"{args.save_dir}/{args.dataset}_{args.model}_{ckpt_name}.json", "w") as f:
            json.dump(class_acc, f, indent=2)
    else:
        # make directory
        os.makedirs(args.data_dir, exist_ok=True)
        ckpt_dir = os.path.join(args.ckpt_dir, args.dataset+"_"+args.model)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # load dataset
        train_dataset, valid_dataset, classes = load_datasets(args.dataset, args.data_dir, args.download_data)
        if args.dataset in ["celeba"]:
            train_loader = train_dataset.get_loader(train=True, reweight_groups=True, \
                            batch_size=args.batch_size, num_workers=4, pin_memory=True)
            valid_loader = valid_dataset.get_loader(train=False, reweight_groups=None, \
                            batch_size=args.batch_size, num_workers=4, pin_memory=True)
        else:
            print(len(train_dataset), len(valid_dataset))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        print("Classes:\n", classes)
            
        # Change the classifier layer
        change_decision_layer(model, len(classes), requires_grad=True)
            
        # train model
        model = model.to(args.device)
        
        if args.optim=="adamw":
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        max_iter = len(train_loader)*args.num_epochs
        if args.lr > args.eta_min_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=args.eta_min_lr)
        else:
            scheduler = None
            
        train_model(train_loader, valid_loader, model, optimizer, scheduler, args.num_epochs, args.device, ckpt_dir)
        
    
    