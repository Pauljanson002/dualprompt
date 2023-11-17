# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import sys
import argparse
import datetime
import random
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
import utils
import clip
import wandb
import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')


def quick_gelu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


class QuickGELU(torch.nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """
    def __init__(self, inplace: bool = False):
        super(QuickGELU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return quick_gelu(input)

def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    
    wandb.init(project="precontinual", name=f"dualprompt_{args.dataset}", config=vars(args),mode="online",tags=["dualprompt","cvpr"])
    args.name = f"dualprompt_{args.dataset}_{args.model.replace('.','_')}"
    data_loader, class_mask,main_buffer , dataset_train = build_continual_dataloader(args)


    print(f"Creating original model: {args.model}")

    act_layer = torch.nn.GELU
    if args.model == "vit_base_patch16_clip_224.openai":
        act_layer = QuickGELU


    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        act_layer=act_layer,
    )

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
        act_layer=act_layer,
    )
    original_model.to(device)
    model.to(device)  

    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False
        
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False
        model.classnames = get_classnames(args)
    text_encodings = clip.tokenize(model.classnames).to(device)
    clip_model, _ = clip.load("ViT-B/16", device="cuda")
    for param in clip_model.parameters():
        param.requires_grad = False
    if args.model == "vit_base_patch16_clip_224.openai":
        original_model.proj = clip_model.visual.proj.float()
        original_model.text_features = clip_model.encode_text(text_encodings).float()
        original_model.logit_scale = clip_model.logit_scale.exp().float()
        model.proj = clip_model.visual.proj.float()
        model.proj.requires_grad = True
        model.text_features = clip_model.encode_text(text_encodings).float()
        model.text_features.requires_grad = False
        model.logit_scale = clip_model.logit_scale.exp().float()
        model.logit_scale.requires_grad = False
    
    print(args)
    args.main_buffer = main_buffer

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                print('No checkpoint found at:', checkpoint_path)
                return
            _ = evaluate_till_now(model, original_model, data_loader, device, 
                                            task_id, class_mask, acc_matrix, args,)
        
        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

    optimizer = create_optimizer(args, model_without_ddp)
    if args.model == "vit_base_patch16_clip_224.openai":
        optimizer.add_param_group({
            "params": [model.proj],
            "lr": args.lr * 1e-2,
        })

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp, original_model,
                    criterion, data_loader, optimizer, lr_scheduler,
                    device, class_mask, args,dataset_train)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_dualprompt':
        from configs.cifar100_dualprompt import get_args_parser
        config_parser = subparser.add_parser('cifar100_dualprompt', help='Split-CIFAR100 DualPrompt configs')
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
    elif config == 'imr_dualprompt':
        from configs.imr_dualprompt import get_args_parser
        config_parser = subparser.add_parser('imr_dualprompt', help='Split-ImageNet-R DualPrompt configs')
    else:
        raise NotImplementedError
        
    # get_args_parser(config_parser)

    # args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
    
    sys.exit(0)