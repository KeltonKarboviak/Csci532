# Lab 4 GPU Convolution

### Linear Version

Run the following commands to compile and run the linear convolution:

```bash
mkdir build
cd build
g++ ../convolution_cpu.cxx -o convolution_cpu --std=c++11
./convolution_cpu 10 10 5 5
```

> Note: the command-line arguments should be in the following format:
```bash
./convolution_cpu <input height: int> <input width: int> <filter height: int> <filter width: int>
```


### Parallel Version

Run the following commands to compile and run 'cuda_device_query.cu' which will print out the CUDA capable devices:

```bash
mkdir build
cd build
nvcc ../cuda_device_query.cu -o cuda_device_query
./cuda_device_query
```

Run the following commands to compile and run the parallel convolution program:

```bash
mkdir build
cd build
nvcc ../convolution_gpu.cu -o convolution_gpu
./convolution_gpu 10 10 5 5
```

> Note: the command-line arguments for the parallel version match exactly with the linear version.


### Benchmarks

Run the following commands to benchmark the linear vs. parallel version:

```bash
python -u test_runner.py
```

It will run each version with all the valid combinations of input and filter sizes below:

```bash
input_sizes = [32x32, 64x64, 128x128, 256x256, 512x512, 1024x1024]
filter_sizes = [4x4, 8x8, 16x16, 32x32, 64x64]
```

This will output a CSV table to stdout that contains the average runtime of 10 runs for each combination of input and filter sizes.

I found with the results I obtained (results.csv) that the below input and filter sizes ran faster on the GPU than on the CPU:

(256, 32), (256, 64), (512, 16), (512, 32), (512, 64), (1024, 8), (1024, 16), (1024, 32), (1024, 64)
