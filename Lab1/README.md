# Lab 1 Linear & Parallel K-Means

### Download the star files

Run the following commands to download and unzip the star files which we need to run the k-means program:

```bash
wget http://milkyway.cs.rpi.edu/milkyway/download/stars.zip
unzip stars.zip
```

### Linear Version

Run the following commands to compile and run the linear k-means program:

```bash
mkdir build
cd build
g++ ../kmeans_linear.cxx --std=c++11 -o kmeans
kmeans 4 ../stars/stars-*
```

### Parallel Version

Run the following commands to compile and run the parallel k-means program:

```bash
mkdir build
cd build
mpicxx ../kmeans_linear.cxx --std=c++11 -o kmeans_parallel
mpirun -np 4 kmeans_parallel 4 ../stars/stars-*
```
