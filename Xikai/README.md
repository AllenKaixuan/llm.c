#### To run the program
First, replace the makefile in main file with this makefile.
- attention loop unroll

    make train_gpt2_attention

    ./train_gpt2_attention
- gelu loop unroll

    make train_gpt2_gelu_unroll

    ./train_gpt2_gelu_unroll
- gelu precision

    make train_gpt2_gelu_precision

    ./train_gpt2_gelu_precision

- gelu memory aligned

    make train_gpt2_gelu_memory

    ./train_gpt_fp32_gelu_memory_aligned.cu