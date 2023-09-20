import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from tqdm import tqdm
from cldatasets import build_continual_dataloader
from engine import *
import models
import utils
import wandb
import warnings

# traindir = os.path.join("", 'train')
# valdir = os.path.join("", 'val')
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])

# train_dataset = datasets.ImageFolder(
#     traindir,
#     transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ]))

# val_dataset = datasets.ImageFolder(
#     valdir,
#     transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize,
#     ]))
IMGNET_PATH = ""
if False:
    print("=> Dummy data is used!")
    train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
    val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
else:
    traindir = os.path.join(IMGNET_PATH, 'train')
    valdir = os.path.join(IMGNET_PATH, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))


print(f"Creating original model: vit")
original_model = create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=100,
    drop_rate=0.0,
    drop_path_rate=0.0,
    drop_block_rate=None,
)

print(f"Creating model: prompt_vit")


model = create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=100,
    drop_rate=0.0,
    drop_path_rate=0.0,
    drop_block_rate=None,
    prompt_length=5,
    embedding_key="cls",
    prompt_init="uniform",
    prompt_pool=True,
    prompt_key=True,
    pool_size=10,
    top_k=1,
    batchwise_prompt=True,
    prompt_key_init="uniform",
    head_type="token",
    use_prompt_mask=True,
    use_g_prompt=True,
    g_prompt_length=5,
    g_prompt_layer_idx=[0, 1],
    use_prefix_tune_for_g_prompt=True,
    use_e_prompt=True,
    e_prompt_layer_idx=[2, 3, 4],
    use_prefix_tune_for_e_prompt=True,
    same_key_value=False,
)
device = "cuda"
original_model.to(device)
model.to(device)  



checkpoint_path = '/home/paulj/projects/dualprompt-pytorch/output/checkpoint_dualprompt_Split-CIFAR100/task10_checkpoint.pth'
print('Loading checkpoint from:', checkpoint_path)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'])

class_mean_set = []
accuracy_history = []
train_loader = DataLoader(train_dataset, batch_size=32)
X = []
y = []
for (img_batch,label) in tqdm(train_loader,total=len(train_loader),desc="Finding mean"):
    img_batch = img_batch.cuda()
    with torch.no_grad():
        out = model.forward_features(img_batch)
        out = out["x"]
        out = out[:, model.total_prompt_len]
        out = F.normalize(out.detach()).cpu().numpy()
    X.append(out)
    y.append(label)
X = np.concatenate(X)
y = np.concatenate(y)
for i in range(1000):
    image_class_mask = (y == i)
    class_mean_set.append(np.mean(X[image_class_mask],axis=0))

test_loader = DataLoader(val_dataset, batch_size=512)
correct , total = 0 , 0
for (img_batch,label) in tqdm(test_loader,total=len(test_loader),desc="Testing"):
    img_batch = img_batch.cuda()
    with torch.no_grad():
        out = model.forward_features(img_batch)
        out = out["x"]
        out = out[:, model.total_prompt_len]
        out = F.normalize(out.detach()).cpu().numpy()
    predictions = []
    for single_image in out:
        distance = single_image - class_mean_set
        norm = np.linalg.norm(distance,ord=2,axis=1)
        pred = np.argmin(norm)
        predictions.append(pred)
    predictions = torch.tensor(predictions)
    correct += (predictions.cpu() == label.cpu()).sum()
    total += label.shape[0]
print(f"Accuracy at {correct/total}")




    