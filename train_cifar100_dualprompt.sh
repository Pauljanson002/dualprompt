#!/bin/bash
set -e
#SBATCH --job-name=dualprompt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch
#SBATCH -w agi1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH --time=14-0
#SBATCH -o %N_%x_%j.out
#SBTACH -e %N_%x_%j.err

# source /data/jaeho/init.sh
# conda activate torch38gpu
# python main.py \
#         cifar100_dualprompt \
#         --model vit_base_patch16_224 \
#         --batch-size 24 \
#         --data-path /local_datasets/ \
#         --output_dir ./output 



# python main.py \
# cars_dualprompt \
# --model vit_base_patch16_clip_224.openai \
# --batch-size 32 \
# --data-path ~/data/ \
# --output_dir ./output 


# python main.py \
# aircraft_dualprompt \
# --model vit_base_patch16_clip_224.openai \
# --batch-size 32 \
# --data-path ~/data/ \
# --output_dir ./output 


# python main.py \
# cub_dualprompt \
# --model vit_base_patch16_clip_224.openai \
# --batch-size 32 \
# --data-path ~/data/ \
# --output_dir ./output \


# python main.py \
# gtsrb_dualprompt \
# --model vit_base_patch16_clip_224.openai \
# --batch-size 32 \
# --data-path ~/data/ \
# --output_dir ./output \

python main.py \
cifar100_dualprompt \
--model vit_base_patch16_clip_224.openai \
--batch-size 64 \
--data-path ~/data/ \
--output_dir ./output 

# python main.py \
# birdsnap_dualprompt \
# --model vit_base_patch16_clip_224.openai \
# --batch-size 128 \
# --data-path ~/data/ \
# --output_dir ./output 


# python main.py \
# cars_dualprompt \
# --model vit_base_patch16_224 \
# --batch-size 32 \
# --data-path ~/data/ \
# --output_dir ./output 

# python main.py \
# cifar100_dualprompt \
# --model vit_base_patch16_224 \
# --batch-size 32 \
# --data-path ~/data/ \
# --output_dir ./output 

# python main.py \
# aircraft_dualprompt \
# --model vit_base_patch16_224 \
# --batch-size 32 \
# --data-path ~/data/ \
# --output_dir ./output 


# python main.py \
# cub_dualprompt \
# --model vit_base_patch16_224 \
# --batch-size 32 \
# --data-path ~/data/ \
# --output_dir ./output \


# python main.py \
# gtsrb_dualprompt \
# --model vit_base_patch16_224 \
# --batch-size 32 \
# --data-path ~/data/ \
# --output_dir ./output \

# python main.py \
# birdsnap_dualprompt \
# --model vit_base_patch16_clip_224.openai \
# --batch-size 64 \
# --data-path ~/data/ \
# --output_dir ./output 