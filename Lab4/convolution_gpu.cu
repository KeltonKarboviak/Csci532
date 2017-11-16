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


void print_matrix(const char *name, float *matrix, int height, int width) {
    printf("%s\n", name);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf(" %5.2f", matrix[width * y + x]);
        }
        printf("\n");
    }
    printf("\n");
}


__global__
static int POSITION(x, y, width) {
    return width * y + x;
}


__global__
void gpu__convolute(float *input, float *filter, float *output, int output_size) {
    // idx = (width) * (y) + (x)
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int width = gridDim.x * blockDim.x
    int idx = POSITION(x, y, width);

    if (idx < output_size) {
        for (int cy = 0; cy < d_filter_height; cy++) {
            for (int cx = 0; cx < d_filter_width; cx++) {
                output[idx] = input[POSITION(x+cx, y+cy, d_input_width)] * filter[POSITION(cx, cy, d_filter_width)];
            }
        }
    }
}


// static int hPOSITION(x, y, width) {
//     return width * y + x;
// }


void usage(char *executable) {
    printf("ERROR, incorrect arguments.\n");
    printf("usage:\n");
    printf("\t %s <input height: int> <input width: int> <filter height: int> <filter width: int>\n", executable);
    exit(1);
}


int main(int argc, char **argv) {
    if (argc != 4) {
        usage(argv[0]);
    }

    // Hard-code for testing output with linear version
    srand48(20171116);

    int input_height = atoi(argv[1]);
    int input_width = atoi(argv[2]);

    int filter_height = atoi(argv[3]);
    int filter_width = atoi(argv[4]);

    int output_height = input_height - filter_height + 1;
    int output_width = input_width - filter_width + 1;

    // Setup input array
    float *cpu__input = (float*) malloc(input_height * input_width * sizeof(float));
    for (int y = 0; y < input_height; y++) {
        for (int x = 0; x < input_width; x++) {
            cpu__input[POSITION(x, y, input_width)] = drand48() * 100;
        }
    }

    // Setup filter array
    float *cpu__filter = (float*) malloc(filter_height * filter_width * sizeof(float));
    for (int y = 0; y < filter_height; y++) {
        for (int x = 0; x < filter_width; x++) {
            cpu__filter[POSITION(x, y, filter_width)] = drand48() * 100;
        }
    }

    // Setup output array
    float *cpu__output = (float*) malloc(output_height * output_width * sizeof(float));
    for (int y = 0; y < output_height; y++) {
        for (int x = 0; x < output_width; x++) {
            cpu__output[POSITION(x, y, output_width)] = 0.0;
        }
    }

    print_matrix("input", input, input_height, input_width);
    print_matrix("filter", filter, filter_height, filter_width);


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


    int block_height = 32.0;
    int block_width = 32.0;

    dim3 dimGrid(ceil(output_width / block_width), ceil(output_height / block_height), 1);
    dim3 dimBlock(block_width, block_height, 1);

    convolute<<<dimGride, dimBlock>>>(gpu__input, gpu__filter, gpu__output, d_output_height * d_output_width);

    // Copy memory for arrays from GPU -> CPU
    cudaErrorCheck( cudaMemcpy(cpu__output, gpu__output, output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost) );


    print_matrix("output", output, output_height, output_width);

    free(cpu__input);  cpu__input  = nullptr;
    free(cpu__filter); cpu__filter = nullptr;
    free(cpu__output); cpu__output = nullptr;

    cudaFree(gpu__input);  gpu__input  = nullptr;
    cudaFree(gpu__filter); gpu__filter = nullptr;
    cudaFree(gpu__output); gpu__output = nullptr;
}
