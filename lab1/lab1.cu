#include "stdio.h"
#include "stdlib.h"
#include <chrono>


__global__ void add(double *a, double *b, double *sum, size_t n) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < n) {
        sum[idx] = a[idx] + b[idx];
        idx += blockDim.x * gridDim.x;
    }
}


int main() {

    size_t Blocks, Threads;
    scanf("%ld %ld", &Blocks, &Threads);

    size_t n;
    scanf("%ld", &n);

    double *a, *b, *c;
    a = (double*)malloc(sizeof(double) * n);
    b = (double*)malloc(sizeof(double) * n);
    c = (double*)malloc(sizeof(double) * n);

    double *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, sizeof(double) * n);
    cudaMalloc((void**)&dev_b, sizeof(double) * n);
    cudaMalloc((void**)&dev_c, sizeof(double) * n);


    for (size_t i = 0; i != n; i++)
        scanf("%lf", &a[i]);
    cudaMemcpy(dev_a, a, sizeof(double) * n, cudaMemcpyHostToDevice);

    for (size_t i = 0; i != n; i++)
        scanf("%lf", &b[i]);
    cudaMemcpy(dev_b, b, sizeof(double) * n, cudaMemcpyHostToDevice);

    

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    add<<<Blocks, Threads>>>(dev_a, dev_b, dev_c, n);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("%f\n", time);
    cudaMemcpy(c, dev_c, sizeof (double) * n, cudaMemcpyDeviceToHost);

    /*
    for (size_t i = 0; i != n; i++)
        printf("%10.10e ", c[i]);
    printf("\n");
    */
    free(a);
    free(b);
    free(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}