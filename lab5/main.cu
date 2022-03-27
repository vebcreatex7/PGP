#include "stdio.h"
#include "stdlib.h"
#include <algorithm>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define N (1 << 24)

#define BLOCKS 1024
#define THREADS 1024

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

__global__ void histogram(uint* hist, const uint* data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offsetx = gridDim.x * blockDim.x;

    for (int i = idx; i < size; i += offsetx)
        atomicAdd(hist + data[i], 1);
}

__global__ void restore(const uint* input, uint* output, uint* pref, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offsetx = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += offsetx)
        output[atomicAdd(pref + input[i], -1) - 1] = input[i];
}

void CountingSort(const uint *input, uint *output, int size) {
    uint *d_input, *d_output;
    CSC(cudaMalloc((void**)&d_input, size * sizeof(uint)));
    CSC(cudaMemcpy(d_input, input, size * sizeof(uint), cudaMemcpyHostToDevice));
    CSC(cudaMalloc((void**)&d_output, size * sizeof(uint)));

    uint* hist;
    CSC(cudaMalloc((void**)&hist, N * sizeof(uint)));
    CSC(cudaMemset(hist, 0, N * sizeof(uint)));
    
    histogram<<<BLOCKS, THREADS>>>(hist, d_input, size);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    thrust::inclusive_scan(thrust::device, hist, hist + N, hist);
    
    restore<<<BLOCKS, THREADS>>>(d_input, d_output, hist, size);
    CSC(cudaGetLastError());

    
    CSC(cudaMemcpy(output, d_output, sizeof(uint) * size, cudaMemcpyDeviceToHost));

    CSC(cudaFree(hist));
    CSC(cudaFree(d_input));
    CSC(cudaFree(d_output));
} 

int main() {
    int n;
    fread(&n, sizeof(int), 1, stdin);

    uint* input = (uint*)malloc(n * sizeof(uint));
    uint* output = (uint*)malloc(n * sizeof(uint));
    fread(input, sizeof(uint), n, stdin);
    
    CountingSort(input, output, n);
    fwrite(output, sizeof(uint), n, stdout);

    free(input);
    free(output);

}
