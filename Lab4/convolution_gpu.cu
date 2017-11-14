extern "C++" {
    #include "stdlib.h"
    #include "stdio.h"
}

#include <cuda.h>
#include <cuda_runtime.h>


__device__ unsigned int
    d_input_height, d_input_width,
    d_filter_height, d_filter_width,
    d_output_height, d_output_width;
    
#define cudaErrorCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void print_matrix(const char *name, float **matrix, int height, int width) {
    printf("%s\n", name);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf(" %5.2f", matrix[y][x]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__
void gpu__convolute(float *input, float *filter, float *output) {
    // idx = (width) * (y) + (x)
    int idx = (gridDim.x * blockDim.x) * (blockDim.y * blockIdx.y + threadIdx.y) + (blockDim.x * blockIdx.x + threadIdx.x);
}

/*void device_setup() {
    
}*/

int main(int n_arguments, char **arguments) {
    int input_height = 16;
    int input_width = 16;

    float **cpu__input = (float**)malloc(sizeof(float*) * input_height);
    for (int y = 0; y < input_height; y++) {
        cpu__input[y] = (float*)malloc(sizeof(float) * input_width);

        for (int x = 0; x < input_width; x++) {
            cpu__input[y][x] = drand48() * 100;
        }
    }

    int filter_height = 5;
    int filter_width = 5;

    float **cpu__filter = (float**)malloc(sizeof(float*) * filter_height);
    for (int y = 0; y < filter_height; y++) {
        cpu__filter[y] = (float*)malloc(sizeof(float) * filter_width);

        for (int x = 0; x < filter_width; x++) {
            cpu__filter[y][x] = drand48() * 100;
        }
    }

    int output_height = input_height - filter_height + 1;
    int output_width = input_width - filter_width + 1;

    float **cpu__output = (float**)malloc(sizeof(float*) * output_height);
    for (int y = 0; y < output_height; y++) {
        cpu__output[y] = (float*)malloc(sizeof(float) * output_width);

        for (int x = 0; x < output_width; x++) {
            cpu__output[y][x] = 0.0;
        }
    }

    print_matrix("input", input, input_height, input_width);
    print_matrix("filter", filter, filter_height, filter_width);

    /*for (int y = 0; y < output_height; y++) {
        for (int x = 0; x < output_width; x++) {

            for (int cy = 0; cy < filter_height; cy++) {
                for (int cx = 0; cx < filter_width; cx++) {
                    cpu__output[y][x] += cpu__input[y + cy][x + cx] * cpu__filter[cy][cx];
                }
            }

        }
    }*/

    
    
    
    
    
    cudaSetDevice(0);
    
    // Copy scalar variables onto GPU
    cudaErrorCheck( cudaMemcpyToSymbol(d_input_height, (void*) &input_height, sizeof(unsigned int)) );
    cudaErrorCheck( cudaMemcpyToSymbol(d_input_width,  (void*) &input_width,  sizeof(unsigned int)) );
    
    cudaErrorCheck( cudaMemcpyToSymbol(d_filter_height, (void*) &filter_height, sizeof(unsigned int)) );
    cudaErrorCheck( cudaMemcpyToSymbol(d_filter_width,  (void*) &filter_width,  sizeof(unsigned int)) );
    
    cudaErrorCheck( cudaMemcpyToSymbol(d_output_height, (void*) &output_height, sizeof(unsigned int)) );
    cudaErrorCheck( cudaMemcpyToSymbol(d_output_width,  (void*) &output_width,  sizeof(unsigned int)) );
    
    float
        *gpu__input,
        *gpu__filter,
        *gpu__output;
    
    
    // Allocate memory for arrays on GPU
    cudaErrorCheck( cudaMalloc((void**) &gpu__input,  input_height  * input_width  * sizeof(float)) );
    cudaErrorCheck( cudaMalloc((void**) &gpu__filter, filter_height * filter_width * sizeof(float)) );
    cudaErrorCheck( cudaMalloc((void**) &gpu__output, output_height * output_width * sizeof(float)) );
    
    // Copy memory for arrays from CPU -> GPU
    cudaErrorCheck( cudaMemcpy(gpu__input,  cpu__input,  input_height  * input_width  * sizeof(float), cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(gpu__filter, cpu__filter, filter_height * filter_width * sizeof(float), cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(gpu__output, cpu__output, output_height * output_width * sizeof(float), cudaMemcpyHostToDevice) );
    
    
    convolute<<<1, 512>>>(gpu__input, gpu__filter, gpu__output);
    
    
    print_matrix("output", output, output_height, output_width);
}
