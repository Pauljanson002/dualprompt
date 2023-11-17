python main.py \
aircraft_dualprompt \
--model vit_base_patch16_224 \
--batch-size 32 \
--data-path ~/data/ \
--output_dir ./output 

python main.py \
cars_dualprompt \
--model vit_base_patch16_224 \
--batch-size 32 \
--data-path ~/data/ \
--output_dir ./output 

python main.py \
cifar100_dualprompt \
--model vit_base_patch16_224 \
--batch-size 32 \
--data-path ~/data/ \
--output_dir ./output 

python main.py \
gtsrb_dualprompt \
--model vit_base_patch16_224 \
--batch-size 32 \
--data-path ~/data/ \
--output_dir ./output 

python main.py \
birdsnap_dualprompt \
--model vit_base_patch16_224 \
--batch-size 32 \
--data-path ~/data/ \
--output_dir ./output 