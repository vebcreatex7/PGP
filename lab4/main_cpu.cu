#include "stdio.h"
#include "stdlib.h"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

double* find_max(double* begin, double* end) {
    double* max = begin;
    double* iter = begin;
    while(iter != end) {
        iter++;
        if (abs(*iter) > abs(*max))
            max = iter;
    }
    return max;
}

void swap(double* a, double* b) {
    double tmp = *a;
    *a = *b;
    *b = tmp;
}

void swap_rows(int i, int j, double* matrix, int n) {
    for (int cols = 0; cols < 2 * n; cols++) 
        swap(&matrix[i + n * cols], &matrix[j + n * cols]);
}

void printMatrix(double* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = n; j < 2 * n; j++) {
            printf("%0.10e ", matrix[i + j * n]);
        }
        printf("\n");
    }
}

void forwardGauss(double* matrix, int n) {
    for (int i = 0; i < n; i++) {   
        double *max = find_max(&matrix[i + i * n], &matrix[n - 1 + i * n]);

        int idx = (max - (matrix  + i * n));
        if (idx != i)
            swap_rows(i, idx, matrix, n);

        for (int k = i + 1; k < n; k++) {
            for (int j = i + 1; j < 2 * n; j++) {
                matrix[k + j * n] -= (matrix[i + j * n] * matrix[k + i * n] / matrix[i + i * n]);
            }
        }
    }
}


void backwardGauss(double* matrix, int n) {
    for (int i = n - 1; i >= 0; i--) {
        for (int k = i - 1; k >= 0; k--) {
            for (int j = i + 1; j < 2 * n; j++) {
                matrix[k + j * n] -= (matrix[i + j * n] * matrix[k + i * n] / matrix[i + i * n]);
            }
        }
    }
}

void reduce(double* matrix, int n) {
    for (int i = 0; i < n; i++) {
        double tmp = matrix[i + i * n];
        for (int j = i; j < 2 * n; j++) {
            matrix[i + j * n] /=  tmp;
        }
    }
}




int main() {
    int n;
    scanf("%d", &n);
    double* matrix = (double*)malloc(sizeof(double) * n * n * 2);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%lf", &matrix[i + j * n]);
        }
        
    }
    
    

    for (int i = 0; i < n; i++) {
         for (int j = n; j < 2 * n; j++) {
            matrix[i + j * n] = (i == j - n) ? 1. : 0.;
        }
    }


   

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    forwardGauss(matrix, n);
    backwardGauss(matrix, n);
    reduce(matrix, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("%f\n", time);
    //printMatrix(matrix, n);
    
    free(matrix);       
}
