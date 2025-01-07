# Team Project: Yurun Chao

This folder contains development work for the Yurun team project. It contains two aspects of functions, `encoder` and `layernorm`.

Since the development is based on the file `train_gpt2.c`, to use the files in this folder, you need to include them in the file `train_gpt2.c`.

There are different kinds of versions of kernel functions in every file. They are all executed, their elapsed time is recorded, and their results are validated with the CPU version. You need to provide the reference output for the validation.

Here are an example of how to modify the file `train_gpt2.c` to use the files in this folder.

```c
#include "cyr/encoder_backward.cuh"

...

float *wte_tmp = (float*)malloc(Vp * C * sizeof(float));
float *wpe_tmp = (float*)malloc(Vp * C * sizeof(float));

// create a copy of the weights for execution on the GPU
memcpy(wte_tmp, grads.wte, V * C * sizeof(float));
memcpy(wpe_tmp, grads.wpe, V * C * sizeof(float));

encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);

encoder_backward_gpu(wte_tmp, wpe_tmp, grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C, V);
```

The output should be like this:

```
encoder_backward_kernel0 time: 0.104490 ms
encoder_backward_kernel1 time: 0.316485 ms
encoder_backward_kernel2 time: 0.116672 ms
```