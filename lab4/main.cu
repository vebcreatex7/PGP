#include "stdio.h"
#include "stdlib.h"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>


#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)


const int xThreads = 32;
const int yThreads = 32;
const int xBlocks = 32;
const int yBlocks = 32;

struct comparator {
    __host__ __device__ bool operator()(double a, double b) {
        return abs(a) < abs(b);
    }
};


void PrintMatrix(double* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = n; j < 2 * n; j++) {
            printf("%0.10e ", matrix[i + j * n]);
        }
        printf("\n");
    }
}



__global__ void SwapRows(double* dev_matrix, int n, int r1, int r2) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    for (int i = idx; i < 2 * n; i += offsetx) {
        double tmp = dev_matrix[r1 + n * i];
        dev_matrix[r1 + n * i] = dev_matrix[r2 + n * i];
        dev_matrix[r2 + n * i] = tmp;
    }
}

__global__ void ForwardGauss(double* dev_matrix, int n, int i) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

    for (int k = idy + i + 1; k < 2 * n; k += offsety)
        for (int j = idx + i + 1; j < n; j += offsetx)
            dev_matrix[k * n + j]  -= (dev_matrix[k * n + i] * dev_matrix[i * n + j] / dev_matrix[i + i * n]);
}

__global__ void BackwardGauss(double* dev_matrix, int n, int i) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
    
    for (int k = idy + i + 1; k < 2 * n; k += offsety)
        for (int j = i - 1 - idx; j >= 0; j -= offsetx)
            dev_matrix[k * n + j]  -= (dev_matrix[k * n + i] * dev_matrix[i * n + j] / dev_matrix[i + i * n]);
}

__global__ void Normalize(double* dev_matrix, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

    for (int i = idy; i < n; i += offsety)
        for (int j = n + idx; j < 2 * n; j += offsetx)
            dev_matrix[i + j * n] /= dev_matrix[i + i * n];
}


void InverseMatrix(double* matrix, int n) {
    double* dev_matrix;
    CSC(cudaMalloc(&dev_matrix, sizeof(double) * 2 * n * n));
    CSC(cudaMemcpy(dev_matrix, matrix, sizeof(double) * 2 * n * n, cudaMemcpyHostToDevice));

    comparator comp;
    for (int i = 0; i < n; i++) {
        thrust::device_ptr<double> thrust_matrix = thrust::device_pointer_cast(dev_matrix);
        thrust::device_ptr<double> max = thrust::max_element(&thrust_matrix[i + i * n], &thrust_matrix[n + i * n], comp);
        int max_idx = max - (thrust_matrix + i * n);
        if (max_idx != i)
            SwapRows<<<dim3(xBlocks), dim3(xThreads)>>>(dev_matrix, n, i, max_idx);
        ForwardGauss<<<dim3(xBlocks, yBlocks), dim3(xThreads,yThreads)>>>(dev_matrix, n, i);
    }

    for (int i = n - 1; i >=0; i--)
        BackwardGauss<<<dim3(xBlocks, yBlocks), dim3(xThreads,yThreads)>>>(dev_matrix, n, i);

    Normalize<<<dim3(xBlocks, yBlocks), dim3(xThreads,yThreads)>>>(dev_matrix, n);

    CSC(cudaMemcpy(matrix, dev_matrix, sizeof(double) * 2 * n * n, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_matrix));
}


int main() {
    int n;
    scanf("%d", &n);
    double* matrix = (double*)malloc(sizeof(double) * n * n * 2);
    
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%lf", &matrix[i + j * n]);
    
    for (int i = 0; i < n; i++)
         for (int j = n; j < 2 * n; j++)
            matrix[i + j * n] = (i == j - n) ? 1. : 0.;

    InverseMatrix(matrix, n);
    PrintMatrix(matrix, n);

    free(matrix);
}