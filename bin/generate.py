for lr in [0.005,0.03]:
    for length in [5]:
        for size in [10]:
            for topk in [5,4]:
                for epochs in [5,50]:
                    script = f"""
python main.py \\
aircraft_dualprompt \\
--model vit_base_patch16_224 \\
--batch-size 32 \\
--data-path ~/data/ \\
--output_dir ./output \\
--length {length} \\
--size {size} \\
--lr {lr} \\
--epochs {epochs} \\
--top_k {topk} \\
        """
                    print(script)