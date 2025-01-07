# Enhanced Versions of softmax, Cross-Entropy, and logit gradients Based on the Original CPU Version and move to GPU version
Copy and replace the Makefile and .cu files to the main folder, then compile and execution
First please read the "quick start" part, do
```bash
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh
make train_gpt2fp32cu
./train_gpt2fp32cu
```
to get model files.

## 1 Versions with Only the Softmax Function Modified (CPU Version)
These versions focus solely on modifying the Softmax function. Each version can be compiled and executed as follows:

- **Version Names:** `train_gpt2_softmax_forward_gpu_vX` (where `X = 1, 2, 3, 4`)
- **Compilation and execution Command:**
```bash
 make train_gpt2_softmax_forward_gpu_vX
./train_gpt2_softmax_forward_gpu_vX
```

## 2 Versions with Both Softmax and Cross-Entropy Functions Modified (CPU Version)
These versions include modifications to both the Softmax and Cross-Entropy functions.

- **Version Names:** `train_gpt2_softmax_cross_forward_gpu_v1`
- **Compilation and execution Command:**
```bash
 make train_gpt2_softmax_cross_forward_gpu_v1
./train_gpt2_softmax_cross_forward_gpu_v1
```
## 3 Kernel Migration Versions Based on the Original GPU Version (GPU Version)
These versions include modifications to both the Softmax, Cross-Entropy and logit gradients functions.

- **Version Names:** `train_gpt2_fp32_softmax_cross_v1`
- **Compilation and execution Command:**
```bash
 make train_gpt2_fp32_softmax_cross_v1
./train_gpt2_fp32_softmax_cross_v1
```

## 4 Test and Verification
To validate the implementation, use the test version as follows:

- **Version Names:** `test_gpt2_fp32_softmax_cross_v1`
- **Compilation and execution Command:**
```bash
 make test_gpt2_fp32_softmax_cross_v1
./test_gpt2_fp32_softmax_cross_v1
```
