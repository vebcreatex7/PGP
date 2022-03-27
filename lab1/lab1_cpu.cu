#include <iostream>
#include "stdio.h"
#include <stdlib.h>
#include <iomanip>


void add(double *a, double *b, double *c, size_t n) {
    for (int i = 0; i != n; i++) {
        c[i] = a[i] + b[i];
    }
}


int main() {
    size_t n;
    scanf("%ld", &n);

    double *a, *b, *c;
    a = (double*)malloc(sizeof(double) * n);
    b = (double*)malloc(sizeof(double) * n);
    c = (double*)malloc(sizeof(double) * n);

    for (size_t i = 0; i != n; i++)
        scanf("%lf", &a[i]);
    for (size_t i = 0; i != n; i++)
        scanf("%lf", &b[i]);

    
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    add(a, b, c, n);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    printf("%f\n", time);

    /*
    for (size_t i = 0; i != n; i++)
        printf("%10.10e", c[i]);
    printf("\n");
    */
    
}