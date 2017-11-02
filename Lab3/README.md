# Lab 3 Parallel Heat Diffusion Simulation

### Linear Version

Run the following commands to download, compile, and run the linear heat diffusion simulation:

```bash
wget -O heat_diffusion.cxx http://people.cs.und.edu/~tdesell/files/heat_diffusion_mpi/heat_diffusion.cxx
mkdir build
cd build
g++ ../heat_diffusion.cxx -o heat_diffusion --std=c++11
./heat_diffusion linear_test 10 10 4 2 10
```

> Note: the command-line arguments should be in the following format:
> ./heat_diffusion <simulation name : string> <height : int> <width : int> <vertical slices : int> <horizontal slices : int> <time steps : int>

### Parallel Version

Run the following commands to compile and run the parallel k-means program:

```bash
mkdir build
cd build
mpicxx ../heat_diffusion_mpi.cxx -o heat_diffusion_mpi --std=c++11
mpirun -np 8 heat_diffusion_mpi mpi_test 10 10 4 2 10
```

> Note: After the MPI specific arguments, the command-line arguments have the same format as the linear version.
