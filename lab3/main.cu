#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)


struct Point {
    int x;
    int y;
};

struct AvgCov {
    float4 avg;
    double inverse_cov[3][3];
    double det_cov;
};

__constant__ AvgCov features[32];

int mod(int i) {
	return i % 3;
}


void inverse(double res[][3], double cov[][3], double det) {
 
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            res[i][j] = ( cov[mod(j + 1)][mod(i + 1)] * cov[mod(j + 2)][mod(i + 2)] -
						  cov[mod(j + 1)][mod(i + 2)] * cov[mod(j + 2)][mod(i + 1)] ) /
						  det;
}

void ComputeAvgCov(const std::vector<Point>& v, AvgCov* ans, uchar4* data, int w, int h) {
    size_t np = v.size();

    // Мат ожидание
    ans->avg = make_float4(0,0,0,0);
    for (int i = 0; i < np; i++) {
        uchar4 p = data[w * v[i].y + v[i].x];
        ans->avg.x += p.x;
        ans->avg.y += p.y;
        ans->avg.z += p.z;
    }
    ans->avg.x /= np;
    ans->avg.y /= np;
    ans->avg.z /= np;

    // Ковариационная матрица
    double cov[3][3] = {0.};
    for (int i = 0; i < np; i++) {
        uchar4 p = data[w * v[i].y + v[i].x];
        double c[3] = {0.};
        c[0] = p.x - ans->avg.x;
        c[1] = p.y - ans->avg.y;
        c[2] = p.z - ans->avg.z;

        for (int k = 0; k < 3; k++)
            for (int j = 0; j < 3; j++)
                cov[k][j] += c[k] * c[j];
    
    }
	for (int k = 0; k < 3; k++)
            for (int j = 0; j < 3; j++)
                cov[k][j] /= np - 1;

    // Детерминант матрицы
	ans->det_cov = 0;
	for (int i = 0; i < 3; i++) {
		ans->det_cov += (cov[0][i] * cov[1][mod(i + 1)] * cov[2][mod(i + 2)] -
						 cov[0][mod(i + 2)] * cov[1][mod(i + 1)] * cov[2][i]);
	}


    // Обратная матрица
    inverse((ans->inverse_cov), cov, ans->det_cov);

}

__device__ double D(uchar4 p, const AvgCov* feature) {
    double tmp[3] = {0.};
    tmp[0] = p.x - feature->avg.x;
    tmp[1] = p.y - feature->avg.y;
    tmp[2] = p.z - feature->avg.z;

    double first[3] = {0.};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            first[i] += ((double)tmp[j] * feature->inverse_cov[j][i]);
    }
    double second = 0.;
    for (int i = 0; i < 3; i++)
        second += (first[i] * tmp[i]);
    
    return (-second - log(abs(feature->det_cov)));
}



__global__ void kernel(uchar4* data, int w, int h, int nc) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;


	for (int y = idy; y < h; y += offsety) {
		for (int x = idx; x < w; x += offsetx) {
			uchar4 p = data[y * w + x];
			double max = D(p, &features[0]);
			int arg = 0;
			for (int class_n = 1; class_n < nc; class_n++) {

				double d = D(p, &features[class_n]);
				if (d > max) {
					max = d;
					arg = class_n;
				}
			}
			data[y * w + x].w = arg;
		}
	}




}
int main() {

    char in[256];
    char out[256];
    scanf("%s", in);
    scanf("%s", out);

    // Считываем классы

    int nc;
    scanf("%d", &nc);
    std::vector<std::vector<Point>> classes(nc);
    for (int i = 0; i< nc; i++) {
        int np;
        scanf("%d", &np);
        classes[i].resize(np);
        for (int j = 0; j < np; j++) {
            Point tmp;
            scanf("%d", &tmp.x);
            scanf("%d", &tmp.y);
            classes[i][j] = tmp;
        }
    }

	


    // Считываем изображение 
	int w, h;
	FILE *fp = fopen(in, "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

    
    
    // Задаем параметры для каждого класса
    AvgCov* host_features = (AvgCov*)malloc(sizeof(AvgCov) * nc);
    for (int i = 0; i < nc; i++)
        ComputeAvgCov(classes[i], &host_features[i], data, w, h);


	uchar4* dev_data;
	CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
	CSC(cudaMemcpy(dev_data, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));
	CSC(cudaMemcpyToSymbol(features, host_features, sizeof(AvgCov) * nc));


    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);    
    kernel<<<dim3(32,32), dim3(32,32)>>>(dev_data, w, h, nc);
     cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("%f\n", time);

	CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));



// Записываем изображение
    fp = fopen(out, "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	cudaFree(dev_data);
	free(host_features);
    free(data);
	return 0;

}