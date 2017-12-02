#include <algorithm>
using std::min;

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

#ifdef __OPENCL__

#include <cstdio>

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


void print_1d(string name, float *input, int input_h, int input_w) {
    cout << "MATRIX '" << name << "'" << endl;
    for (int y = 0; y < input_h; y++) {
        for (int x = 0; x < input_w; x++) {
            cout << setw(10) << fixed << input[input_w * y + x];
        }
        cout << endl;
    }
}


bool equal_1d(float *X, float *Y, int size) {
    for (int i = 0; i < size; i++) {
        if (X[i] != Y[i])
            return false;
    }

    return true;
}


int POSITION(int y, int x, int width) {
    return width * y + x;
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


void matrix_mul(float *A, float *B, float *C, int C_h, int C_w, int A_w) {
    float result;
    for (int y = 0; y < C_h; y++) {
        for (int x = 0; x < C_w; x++) {
            result = 0.0;
            for (int i = 0; i < A_w; i++) {
                result += A[POSITION(y, i, A_w)] * B[POSITION(i, x, C_w)];
            }
            C[POSITION(y, x, C_w)] = result;
        }
    }
}


void initialize_opencl(int A_size, int B_size, int C_size, int shared_side_length, int tile_width) {
    // OpenCL structures
    cl_int err;

    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_error(err, "couldn't create a context, err: %d", err);

    //Create a command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    check_error(err, "couldn't create a command queue: %d", err);

    // Create Vector of replacements for Kernel string
    vector<string> replacements;
    replacements.push_back("TILE_DEF");
    replacements.push_back(std::to_string(tile_width));

    // Build program
    matrix_mul_program = build_program(context, device, MATRIX_MUL_KERNEL_FILE, replacements);

    // Create a kernel
    kernel = clCreateKernel(matrix_mul_program, "matrix_mul_tiled", &err);
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

    err = clSetKernelArg(kernel, 3, sizeof(int), &shared_side_length);
    check_error(err, "couldn't create width argument: %d", err);
}


void matrix_mul_opencl(float *A, float *B, float *C, int A_size, int B_size, int C_size) {
    cl_int err;

    err = clEnqueueWriteBuffer(queue, A_opencl, CL_TRUE, 0, sizeof(float) * A_size, A, 0, NULL, NULL);
    check_error(err, "couldn't write to the A_opencl buffer: %d", err);

    err = clEnqueueWriteBuffer(queue, B_opencl, CL_TRUE, 0, sizeof(float) * B_size, B, 0, NULL, NULL);
    check_error(err, "couldn't write to the B_opencl buffer: %d", err);

    // Enqueue kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    check_error(err, "couldn't enqueue the kernel: %d", err);

    err = clFinish(queue);
    check_error(err, "queue errored on finish: %d", err);

    // Read the kernel's output
    err = clEnqueueReadBuffer(queue, C_opencl, CL_TRUE, 0, sizeof(float) * C_size, C, 0, NULL, NULL);
    check_error(err, "couldn't read from the C_opencl buffer: %d", err);
}


void usage(char* executable) {
    cerr << "ERROR, incorrect arguments." << endl
         << "usage:" << endl
         << "\t " << executable << " <A height: int> <A width: int> <B height: int> <B width: int>" << endl
         << "Note: <A width> must equal <B height>"
         << endl;
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

    if (A_w != B_h) {
        usage(argv[0]);
    }

    float *A = (float*) malloc(sizeof(float) * A_size);
    float *B = (float*) malloc(sizeof(float) * B_size);
    float *C_cpu = (float*) malloc(sizeof(float) * C_size);
    float *C_gpu = (float*) malloc(sizeof(float) * C_size);

    for (int i = 0; i < A_size; i++) {
        A[i] = i;  // drand48();
    }

    for (int i = 0; i < B_size; i++) {
        B[i] = i;  // drand48();
    }

    for (int i = 0; i < C_size; i++) {
        C_cpu[i] = 0.0;
        C_gpu[i] = 0.0;
    }

    print_1d("A", A, A_h, A_w);
    print_1d("B", B, B_h, B_w);

    cout << endl << "created initial arrays." << endl;


    cout << endl << "initializing opencl." << endl;

    // Create device
    device = create_device();

    size_t max_work_item_sizes[3];
    size_t max_group_size;
    cl_ulong local_memory_limit;

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), &max_work_item_sizes, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_group_size), &max_group_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_memory_limit), &local_memory_limit, NULL);

    // Allocate arrays for holding Kernel dimensions
    global_size = (size_t*) malloc(sizeof(size_t) * 2);
    local_size = (size_t*) malloc(sizeof(size_t) * 2);

    // Create sizes for Kernel
    global_size[0] = C_h;  // y-dim
    global_size[1] = C_w;  // x-dim

    local_size[0] = 1;  // Block y-dim
    local_size[1] = 1;  // Block x-dim

    int i = 1;
    int smaller = min(C_h, C_w);
    int optimal = smaller / i++;

    while (optimal != 1) {
        if (C_h % optimal == 0 && C_w % optimal == 0
            && optimal <= max_work_item_sizes[0] && optimal <= max_work_item_sizes[1]
            && optimal * optimal * sizeof(float) <= local_memory_limit
            && optimal * optimal <= max_group_size
        ) {
                break;
        }

        optimal = smaller / i++;
    }

    local_size[0] = local_size[1] = optimal;


    printf("Max Work Item Sizes: (%lu, %lu)\n", max_work_item_sizes[0], max_work_item_sizes[1]);
    printf("Local Memory Limit: %llu\n", local_memory_limit);
    printf("Max Group Size: %lu\n", max_group_size);
    printf("Global Size: (%lu, %lu)\n", global_size[0], global_size[1]);
    printf("Local Size: (%lu, %lu)\n", local_size[0], local_size[1]);

    initialize_opencl(A_size, B_size, C_size, A_w, local_size[0]);

    cout << "initialized successfully." << endl;


    using namespace std::chrono;

    high_resolution_clock::time_point t1, t2;
    duration<float, std::milli> time_span;

    t1 = high_resolution_clock::now();
    matrix_mul_opencl(A, B, C_gpu, A_size, B_size, C_size);
    t2 = high_resolution_clock::now();

    time_span = t2 - t1;

    cout << endl << "OpenCL Matrix Mul took: " << time_span.count() / 1000.0 << " seconds." << endl << endl;

    print_1d("C_GPU", C_gpu, C_h, C_w);

    t1 = high_resolution_clock::now();
    matrix_mul(A, B, C_cpu, C_h, C_w, A_w);
    t2 = high_resolution_clock::now();

    time_span = t2 - t1;

    cout << endl << "CPU Matrix Mul took: " << time_span.count() / 1000.0 << " seconds." << endl << endl;

    print_1d("C_CPU", C_cpu, C_h, C_w);

    cout << endl << "Matrices equal? " << std::boolalpha << equal_1d(C_gpu, C_cpu, C_size) << endl;

    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);

    free(global_size);
    free(local_size);

    clReleaseMemObject(A_opencl);
    clReleaseMemObject(B_opencl);
    clReleaseMemObject(C_opencl);

    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
}
