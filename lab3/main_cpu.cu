#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>

#define ull unsigned long long

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
    double avg[3];
    double inverse_cov[3][3];
    double det_cov;
};



AvgCov features[32];



void inverse(double res[][3], double cov[][3], double det) {
    double M[3][3];

    M[0][0] = cov[1][1] * cov[2][2] - cov[1][2] * cov[2][1];
    M[0][1] = -(cov[1][0] * cov[2][2] - cov[1][2] * cov[2][0]);
    M[0][2] = cov[1][0] * cov[2][1] - cov[1][1] * cov[2][0];
    M[1][0] = -(cov[0][1] * cov[2][2] - cov[0][2] * cov[2][1]);
    M[1][1] = cov[0][0] * cov[2][2] - cov[0][2] * cov[2][0];
    M[1][2] = -(cov[0][0] * cov[2][1] - cov[0][1] * cov[2][0]);
    M[2][0] = cov[0][1] * cov[1][2] - cov[0][2] * cov[1][1];
    M[2][1] = -(cov[0][0] * cov[1][2] - cov[0][2] * cov[1][0]);
    M[2][2] = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            res[i][j] = M[j][i] / det;
        
            
}



AvgCov ComputeAvgCov(const std::vector<Point>& v, uchar4* data, int w, int h) {
    ull np = v.size();
    AvgCov ans;

    // Мат ожидание
    ans.avg[3] = {.0};
    for (int i = 0; i < np; i++) {
        uchar4 p = data[w * v[i].y + v[i].x];
        ans.avg[0] += double(p.x) / np;
        ans.avg[1] += double(p.y) / np;
        ans.avg[2] += double(p.z) / np;
    }

    // Ковариационная матрица
    double cov[3][3] = {{0.}, {0.}, {0.}};
    for (int i = 0; i < np; i++) {
        uchar4 p = data[w * v[i].y + v[i].x];
        double c[3];
        c[0] = double(p.x) - ans.avg[0];
        c[1] = double(p.y) - ans.avg[1];
        c[2] = double(p.z) - ans.avg[2];

        cov[0][0] += (c[0] * c[0] / (np - 1));
        cov[1][0] += (c[1] * c[0] / (np - 1));
        cov[2][0] += (c[2] * c[0] / (np - 1));

        cov[0][1] += (c[0] * c[1] / (np - 1));
        cov[1][1] += (c[1] * c[1] / (np - 1));
        cov[2][1] += (c[2] * c[1] / (np - 1));

        cov[0][2] += (c[0] * c[2] / (np - 1));
        cov[1][2] += (c[1] * c[2] / (np - 1));
        cov[2][2] += (c[2] * c[2] / (np - 1));    
    }

    // Детерминант матрицы
    ans.det_cov = (cov[0][0] * cov[1][1] * cov[2][2] + cov[0][1] * cov[1][2] * cov[2][0] + cov[1][0] * cov[2][1] * cov[0][2]) -
                    (cov[0][2] * cov[1][1] * cov[2][0] + cov[1][0] * cov[0][1] * cov[2][2] + cov[1][2] * cov[2][1] * cov[0][0]);


    // Обратная матрица
    inverse(ans.inverse_cov, cov, ans.det_cov);

    return ans;

}

double D(uchar4 p, int j) {
    double tmp[3] = {.0};
    tmp[0] = double(p.x) - features[j].avg[0];
    tmp[1] = double(p.y) - features[j].avg[1];
    tmp[2] = double(p.z) - features[j].avg[2];

    double first = tmp[0] * (tmp[0] * features[j].inverse_cov[0][0] + tmp[1] * features[j].inverse_cov[1][0] + tmp[2] * features[j].inverse_cov[2][0]) + 
                   tmp[1] * (tmp[0] * features[j].inverse_cov[0][1] + tmp[1] * features[j].inverse_cov[1][1] + tmp[2] * features[j].inverse_cov[2][1]) +
                   tmp[2] * (tmp[0] * features[j].inverse_cov[0][2] + tmp[0] * features[j].inverse_cov[1][2] + tmp[2] * features[j].inverse_cov[2][2]);

    return (-first - log(abs(features[j].det_cov)));
}

void kernel(uchar4* data, uchar4* res, int w, int h, int np) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uchar4 p = data[y * w + x];
            double max = D(p, 0);
            printf("max %lf\n", max);
            int arg = 0;
            for (int j = 1; j < np; j++) {
                double d = D(p, j);
                printf("d = %lf\n", d);
                if (d > max) {
                   
                    max = d;
                    arg = j;
                }
            }
            printf("arg");
            p.w = (unsigned char)arg;
            res[y * w + x] = p;
        }
    }
}


int main() {
    char in[256];
    char out[256];
    scanf("%s", in);
    scanf("%s", out);

    // Считываем классы 
    int np;
    scanf("%d", &np);
    std::vector<std::vector<Point>> classes(np);
    for (int i = 0; i< np; i++) {
        ull n;
        scanf("%llu", &n);
        classes[i].resize(n);
        for (int j = 0; j < n; j++) {
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
    for (int i = 0; i < np; i++) {
        features[i] = ComputeAvgCov(classes[i], data, w, h);
        printf("%lf %lf %lf\n", features[i].avg[0], features[i].avg[1], features[i].avg[2]);
        for (int k = 0; k < 3; k++) {
            for (int j = 0; j < 3; j++) {
                printf("%lf ", features[i].inverse_cov[k][j]);
            }
            printf("\n");
        }
        printf("%lf\n", features[i].det_cov);
    }
        


    
    


    uchar4 *res = (uchar4 *)malloc(sizeof(uchar4) * w * h);

   // kernel(data, res, w, h, np);

    
    // Записываем изображение
    fp = fopen(out, "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(res, sizeof(uchar4), w * h, fp);
	fclose(fp);

    free(res);
    free(data);
	return 0;
}