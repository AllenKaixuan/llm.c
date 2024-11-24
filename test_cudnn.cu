#include <cudnn.h>
#include <stdio.h>

int main() {
    cudnnHandle_t handle;

    // Initialize cuDNN
    cudnnStatus_t status = cudnnCreate(&handle);
    if (status == CUDNN_STATUS_SUCCESS) {
        printf("cuDNN initialized successfully.\n");

        // Print cuDNN version
        printf("cuDNN version: %d\n", CUDNN_VERSION);

        // Destroy cuDNN handle
        cudnnDestroy(handle);
        return 0;
    } else {
        printf("Failed to initialize cuDNN: %s\n", cudnnGetErrorString(status));
        return -1;
    }
}
