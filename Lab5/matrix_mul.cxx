#include <cmath>
using std::sqrt;

#include <chrono>

#include <iomanip>
using std::fixed;
using std::setw;
using std::setprecision;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <numeric>

#ifdef __OPENCL__

#include <cstdio>
#include <cstring>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "opencl_utils.hxx"

#endif


//set up global opencl variables
cl_device_id device;
cl_context context;
cl_program matrix_mul_program;
cl_kernel kernel;
cl_command_queue queue;

cl_mem A_opencl;
cl_mem B_opencl;
cl_mem C_opencl;

size_t *global_size;
size_t *local_size;

#define MATRIX_MUL_KERNEL_FILE "../matrix_mul_kernel.cl"


//initialize the memory for a 3D array
void initialize_3d(float ****v, uint32_t v_z, uint32_t v_y, uint32_t v_x) {
    (*v) = (float***)malloc(sizeof(float**) * v_z);
    for (uint32_t z = 0; z < v_z; z++) {
        (*v)[z] = (float**)malloc(sizeof(float*) * v_y);
        for (uint32_t y = 0; y < v_y; y++) {
            (*v)[z][y] = (float*)malloc(sizeof(float) * v_x);
        }
    }
}

//set the values in a 3d array to random numbers
void set_to_random_3d(float ***v, uint32_t v_z, uint32_t v_y, uint32_t v_x) {
    for (uint32_t z = 0; z < v_z; z++) {
        for (uint32_t y = 0; y < v_y; y++) {
            for (uint32_t x = 0; x < v_x; x++) {
                v[z][y][x] = drand48();
            }
        }
    }
}

//set the values in a 3d array to 0
void set_to_zero_3d(float ***v, uint32_t v_z, uint32_t v_y, uint32_t v_x) {
    for (uint32_t z = 0; z < v_z; z++) {
        for (uint32_t y = 0; y < v_y; y++) {
            for (uint32_t x = 0; x < v_x; x++) {
                v[z][y][x] = 0.0;
            }
        }
    }
}

//copy the values from a 3d array to a flattened 1d array
void copy_3d_to_1d(float ***input, uint32_t input_z, uint32_t input_y, uint32_t input_x, float *output) {
    uint32_t current_output = 0;
    for (uint32_t z = 0; z < input_z; z++) {
        for (uint32_t y = 0; y < input_y; y++) {
            for (uint32_t x = 0; x < input_x; x++) {
                output[current_output++] = input[z][y][x];
            }
        }
    }
}

//copy the values from a flattened 1d array to a 3d array
void copy_1d_to_3d(float *input, uint32_t output_z, uint32_t output_y, uint32_t output_x, float ***output) {
    uint32_t current_output = 0;
    for (uint32_t z = 0; z < output_z; z++) {
        for (uint32_t y = 0; y < output_y; y++) {
            for (uint32_t x = 0; x < output_x; x++) {
                output[z][y][x] = input[current_output++];
            }
        }
    }
}


void print_1d(string name, float *input, int input_h, int input_w) {
    cout << "MATRIX '" << name << "'" << endl;
    for (int y = 0; y < input_h; y++) {
        for (int x = 0; x < input_w; x++) {
            cout << setw(10) << fixed << input[input_w * y + x];
        }
        cout << endl;
    }
}


void print_3d(string name, float ***input, uint32_t input_z, uint32_t input_y, uint32_t input_x) {
    cout << "MATRIX '" << name << "'" << endl;
    for (uint32_t z = 0; z < input_z; z++) {
        for (uint32_t y = 0; y < input_y; y++) {
            for (uint32_t x = 0; x < input_x; x++) {
                cout << setw(10) << fixed << input[z][y][x];
            }
            cout << endl;
        }
        cout << endl;
    }
}


void matrix_add(float ***A, float ***B, float ***C, int input_z, int input_y, int input_x) {
    for (uint32_t z = 0; z < input_z; z++) {
        for (uint32_t y = 0; y < input_y; y++) {
            for (uint32_t x = 0; x < input_x; x++) {
                C[z][y][x] = A[z][y][x] + B[z][y][x];
            }
        }
    }
}


void matrix_mul(float *A, float *B, float *C, int C_h, int C_w) {

}


bool equal_3d(float ***M1, float ***M2, int input_z, int input_y, int input_x) {
    for (uint32_t z = 0; z < input_z; z++) {
        for (uint32_t y = 0; y < input_y; y++) {
            for (uint32_t x = 0; x < input_x; x++) {
                if (M1[z][y][x] != M2[z][y][x]) return false;
            }
        }
    }
    return true;
}

void initialize_opencl(int A_size, int B_size, int C_size) {
    //OpenCL structures
    cl_int err;

    //Create device and context
    device = create_device();

    size_t maxWorkItemSizes[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), &maxWorkItemSizes, NULL);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_error(err, "couldn't create a context, err: %d", err);

    //Create a command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    check_error(err, "couldn't create a command queue: %d", err);

    // Build program
    matrix_mul_program = build_program(context, device, MATRIX_MUL_KERNEL_FILE);

    // Create a kernel
    kernel = clCreateKernel(matrix_mul_program, "matrix_mul", &err);
    check_error(err, "couldn't create a kernel: %d", err);

    //A_opencl, B_opencl, and C_opencl are set as global variables so we can reuse them
    A_opencl = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * A_size, NULL, &err);
    check_error(err, "could not create A_opencl buffer: %d", err);

    B_opencl = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * B_size, NULL, &err);
    check_error(err, "could not create B_opencl buffer: %d", err);

    C_opencl = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * C_size, NULL, &err);
    check_error(err, "could not create C_opencl buffer: %d", err);

    // only need to set the kernel arguments once, and we can then re-use them if
    // those cl_mem variables don't change
    // Create kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A_opencl);
    check_error(err, "couldn't create A_opencl argument: %d", err);

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &B_opencl);
    check_error(err, "couldn't create B_opencl argument: %d", err);

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &C_opencl);
    check_error(err, "couldn't create C_opencl argument: %d", err);

    global_size = (size_t*) malloc(sizeof(size_t) * 2);
    local_size = (size_t*) malloc(sizeof(size_t) * 2);
}


void matrix_mul_opencl(float *A, float *B, float *C, int A_size, int B_size, int C_size) {
    cl_int err;

    // int size = sizeof(float) * z * y * x;

    err = clEnqueueWriteBuffer(queue, A_opencl, CL_TRUE, 0, A_size, A, 0, NULL, NULL);
    check_error(err, "couldn't write to the A_opencl buffer: %d", err);

    err = clEnqueueWriteBuffer(queue, B_opencl, CL_TRUE, 0, B_size, B, 0, NULL, NULL);
    check_error(err, "couldn't write to the B_opencl buffer: %d", err);

    // Enqueue kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    check_error(err, "couldn't enqueue the kernel: %d", err);

    err = clFinish(queue);
    check_error(err, "queue errored on finish: %d", err);

    // Read the kernel's output
    err = clEnqueueReadBuffer(queue, C_opencl, CL_TRUE, 0, C_size, C, 0, NULL, NULL);
    check_error(err, "couldn't read from the C_opencl buffer: %d", err);
}


void usage(char* executable) {
    cerr << endl;
    exit(1);
}


int main(int argc, char **argv) {
    if (argc != 5) {
        usage(argv[0]);
    }

    int A_h = atoi(argv[1]);
    int A_w = atoi(argv[2]);
    int A_size = A_h * A_w;

    int B_h = atoi(argv[3]);
    int B_w = atoi(argv[4]);
    int B_size = B_h * B_w;

    int C_h = A_h;
    int C_w = B_w;
    int C_size = C_h * C_w;

    float *A = (float*) malloc(sizeof(float) * A_size);
    float *B = (float*) malloc(sizeof(float) * B_size);
    float *C = (float*) malloc(sizeof(float) * B_size);
    // float *C_gpu;

    for (int i = 0; i < A_size; i++) {
        A[i] = i;  // drand48();
    }

    for (int i = 0; i < B_size; i++) {
        B[i] = i;  // drand48();
    }

    for (int i = 0; i < C_size; i++) {
        C[i] = 0.0;
    }

    print_1d("A", A, A_h, A_w);
    print_1d("B", B, B_h, B_w);

    cout << "created initial arrays." << endl;

    cout << endl << "initializing opencl." << endl;
    initialize_opencl(A_size, B_size, C_size);
    cout << "initialized successfully." << endl;


    // Create sizes for Kernel

    global_size[0] = C_h;  // y-dim
    global_size[1] = C_w;  // x-dim

    // TODO: need to figure out the optimal size for the blocks



    using namespace std::chrono;

    high_resolution_clock::time_point t1, t2;
    duration<float, std::milli> time_span;

    t1 = high_resolution_clock::now();
    matrix_mul_opencl(A, B, C, A_size, B_size, C_size);
    t2 = high_resolution_clock::now();

    time_span = t2 - t1;

    cout << "OpenCL Matrix Mul took: " << time_span.count() / 1000.0 << " seconds." << endl << endl;

    // copy_1d_to_3d(C_flat, z, y, x, C_gpu);
    // print_1d("C_GPU", C, C_h, C_w);

    t1 = high_resolution_clock::now();
    matrix_mul(A, B, C, C_h, C_w);
    t2 = high_resolution_clock::now();

    time_span = t2 - t1;

    cout << "CPU Matrix Mul took: " << time_span.count() / 1000.0 << " seconds." << endl << endl;

    // cout << "Matrices equal? " << equal_3d(C_gpu, C, z, y, x) << endl;

    clReleaseMemObject(A_opencl);
    clReleaseMemObject(B_opencl);
    clReleaseMemObject(C_opencl);

    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
}
