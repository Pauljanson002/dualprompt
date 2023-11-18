import sys
import argparse
import datetime
import random
from typing import Any
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from cldatasets import build_continual_dataloader,get_classnames
from engine import *
import models
from torchvision import transforms
import utils
import wandb
import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')
import clip
from torchvision.datasets import ImageFolder
def quick_gelu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


class QuickGELU(torch.nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """
    def __init__(self, inplace: bool = False):
        super(QuickGELU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return quick_gelu(input)


class ImageNet(ImageFolder):
    def __init__(self,transform):
        root = '/home/paulj/data/imagenet/val'
        super().__init__(root,transform)
        folder_to_classes = {}
        with open('/home/paulj/data/imagenet/imagenet_class_name.txt','r') as f:
            for line in f.readlines():
                folder,_,class_name = line.strip().split()
                folder_to_classes[folder] = class_name
        for i,folder in enumerate(self.classes):
            self.classes[i] = folder_to_classes[folder]





def main(args):
    utils.init_distributed_mode(args)

    device = torch.device("cuda")
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args.model = "vit_base_patch16_clip_224.openai"

    cudnn.benchmark = True
    wandb.init(project="precontinual", name=f"imagenet_dualprompt_{args.dataset}_{args.model}", config=vars(args),mode="disabled",tags=["cvpr","imagenet-dualprompt"])
    args.name = f"imagenet_dualprompt_{args.dataset}_{args.model.replace('.','_')}"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    dataset = ImageNet(transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)
    text_encodings = clip.tokenize(dataset.classes).to(device)

    clip_model, _ = clip.load("ViT-B/16", device="cuda")

    # turn off gradients for CLIP
    for param in clip_model.parameters():
        param.requires_grad = False
    text_features = clip_model.encode_text(text_encodings).float()
    logit_scale = clip_model.logit_scale.exp().float()
    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        act_layer=QuickGELU,
    )
    original_model.to(device)
    # Turn off gradients for original model
    for param in original_model.parameters():
        param.requires_grad = False
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
        act_layer=QuickGELU,
    )
    model.to(device) 
    model.classnames = dataset.classes
    model.text_features = text_features
    model.logit_scale = logit_scale
    model.proj = clip_model.visual.proj.float()
    checkpoint_path = f"/home/paulj/projects/dualprompt-pytorch/output/checkpoint_dualprompt_{args.dataset}_vit_base_patch16_clip_224_openai/task10_checkpoint.pth"
    print('Loading checkpoint from:', checkpoint_path)
    checkpoint = torch.load(checkpoint_path,map_location="cuda")
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # Turn off gradients for model
    for param in model.parameters():
        param.requires_grad = False
    from tqdm import tqdm
    correct = 0
    total = 0
    pbar = tqdm(enumerate(data_loader),total=len(data_loader),desc="eval")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            cls_features = original_model(inputs)["pre_logits"]
            output = model(inputs,cls_features=cls_features)
            logits = output['logits']
            preds = logits.argmax(dim=1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()
            pbar.set_description(f"eval accuracy: {correct/total * 100:.2f}")

    final_accuracy = (correct / total) * 100
    print("Final accuracy:",final_accuracy)




















if __name__ == '__main__':
    parser = argparse.ArgumentParser('dualprompt training and evaluation configs')
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_dualprompt':
        from configs.cifar100_dualprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_dualprompt', help='Split-CIFAR100 dualprompt configs')
        get_args_parser(config_parser)
        args = parser.parse_args()
        args.num_tasks = 10
        args.memory = 2000
        args.nb_classes = 100

    elif config == "cub_dualprompt":
        from configs.cub_dualprompt import get_args_parser
        config_parser = subparser.add_parser('cub_dualprompt', help='CUB-200-2011 dualprompt configs')
        get_args_parser(config_parser)
        args = parser.parse_args()
        args.num_tasks = 10
        args.memory = 240
        args.nb_classes = 200
    elif config == "cars_dualprompt":
        from configs.cars_dualprompt import get_args_parser
        config_parser = subparser.add_parser("cars_dualprompt", help="Stanford Cars dualprompt configs")
        get_args_parser(config_parser)
        args = parser.parse_args()
        args.num_tasks = 10
        args.memory = 240
        args.nb_classes = 190
    elif config == "aircraft_dualprompt":
        from configs.aircraft_dualprompt import get_args_parser
        config_parser = subparser.add_parser("aircraft_dualprompt", help="FGVC-Aircraft dualprompt configs")
        get_args_parser(config_parser)
        args = parser.parse_args()
        args.num_tasks = 10
        args.memory = 250
        args.nb_classes = 100
    elif config == "country_dualprompt":
        from configs.country_dualprompt import get_args_parser
        config_parser = subparser.add_parser("country_dualprompt", help="Country dualprompt configs")
        get_args_parser(config_parser)
        args = parser.parse_args()
        args.num_tasks = 10
        args.memory = 1000
        args.nb_classes = 200
    elif config == "gtsrb_dualprompt":
        from configs.gtsrb_dualprompt import get_args_parser
        config_parser = subparser.add_parser("gtsrb_dualprompt", help="GTSRB dualprompt configs")
        get_args_parser(config_parser)
        args = parser.parse_args()
        args.num_tasks = 10
        args.memory = 1000
        args.nb_classes = 40
    elif config == "birdsnap_dualprompt":
        from configs.birdsnap_dualprompt import get_args_parser
        config_parser = subparser.add_parser("birdsnap_dualprompt", help="Birdsnap dualprompt configs")
        get_args_parser(config_parser)
        args = parser.parse_args()
        args.num_tasks = 10
        args.memory = 1500
        args.nb_classes = 500
    elif config == 'five_datasets_dualprompt':
        from configs.five_datasets_dualprompt import get_args_parser
        config_parser = subparser.add_parser('five_datasets_dualprompt', help='5-Datasets dualprompt configs')
    else:
        raise NotImplementedError
    
    # get_args_parser(config_parser)

    # args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

    sys.exit(0)