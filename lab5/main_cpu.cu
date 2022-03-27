#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>

#define N 1<<24

void Hist(int* data, int* count, int n) {
    for (int i = 0; i < n; i++) {
        ++count[data[i]];
    }
}

void Scan(int *count,  int max) {
    for (int i = 1; i <= max; i++)
        count[i] += count[i - 1];
}

void CountingSort(int* input, int* output, int n, int max) {

    int* count = (int*)malloc((max + 1) * sizeof(int));
    memset(count, 0, (max + 1) * sizeof(int));
    
    Hist(input, count, n);
    Scan(count, max);
    
    for (int i = 0; i < n; i++) {
        output[count[input[i]] - 1] = input[i];
        --count[input[i]];
    }

    free(count);
}

int main() {
    int n;
    fread(&n, sizeof(int), 1, stdin);

    int* input = (int*)malloc(n * sizeof(int));
    int* output = (int*)malloc(n * sizeof(int));
    fread(input, sizeof(int), n, stdin);

    int max = *std::max_element(input, input + n);

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    CountingSort(input, output, n, max);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("%f\n", time);
    
    //fwrite(output, sizeof(int), n, stdout);
    //for (int i = 0; i < n; i++)
    //    printf("%d ", output[i]);

    
    free(input);
    free(output);
}