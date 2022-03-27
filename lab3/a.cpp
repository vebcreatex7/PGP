#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <string.h>

#define ull unsigned long long

struct rgbw {
    unsigned char x;
    unsigned char y;
    unsigned char z;
    unsigned char w;
};

typedef rgbw uchar4;

struct Point {
    int x;
    int y;
};



struct AvgCov {
    int avg[3];
    double inverse_cov[3][3];
    double det_cov;
};




AvgCov features[32];



void inverse(double res[][3], double cov[][3], double det) {
    double M[3][3] = {0.};

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



void ComputeAvgCov(const std::vector<Point>& v, AvgCov* ans, uchar4* data, int w, int h) {
    ull np = v.size();


    // Мат ожидание
    ans->avg[3] = {0};
    for (int i = 0; i < np; i++) {
        uchar4 p = data[w * v[i].y + v[i].x];
        ans->avg[0] += (p.x);
        ans->avg[1] += (p.y);
        ans->avg[2] += (p.z);
    }
    ans->avg[0] /= np;
    ans->avg[1] /= np;
    ans->avg[2] /= np;

    // Ковариационная матрица
    double cov[3][3] = {0.};
    for (int i = 0; i < np; i++) {
        uchar4 p = data[w * v[i].y + v[i].x];
        int c[3] = {0};
        c[0] = (p.x) - ans->avg[0];
        c[1] = (p.y) - ans->avg[1];
        c[2] = (p.z) - ans->avg[2];

        for (int k = 0; k < 3; k++)
            for (int j = 0; j < 3; j++)
                cov[k][j] += (c[k] * c[j]);
        
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < 3; j++)
                cov[k][j] /= (np - 1);

    /*
        cov[0][0] += (c[0] * c[0] / (np - 1));
        cov[1][0] += (c[1] * c[0] / (np - 1));
        cov[2][0] += (c[2] * c[0] / (np - 1));

        cov[0][1] += (c[0] * c[1] / (np - 1));
        cov[1][1] += (c[1] * c[1] / (np - 1));
        cov[2][1] += (c[2] * c[1] / (np - 1));

        cov[0][2] += (c[0] * c[2] / (np - 1));
        cov[1][2] += (c[1] * c[2] / (np - 1));
        cov[2][2] += (c[2] * c[2] / (np - 1));
    */
    }

    // Детерминант матрицы
    ans->det_cov = ((cov[0][0] * cov[1][1] * cov[2][2]) + (cov[0][1] * cov[1][2] * cov[2][0]) + (cov[1][0] * cov[2][1] * cov[0][2])) -
                    ((cov[0][2] * cov[1][1] * cov[2][0]) + (cov[1][0] * cov[0][1] * cov[2][2]) + (cov[1][2] * cov[2][1] * cov[0][0]));


    // Обратная матрица
    inverse(ans->inverse_cov, cov, ans->det_cov);


}



double D(uchar4 p, AvgCov* feature) {
    int tmp[3] = {0};
    tmp[0] = p.x - feature->avg[0];
    tmp[1] = p.y - feature->avg[1];
    tmp[2] = p.z - feature->avg[2];

    double first[3] = {0.};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++)
            first[i] += ((double)tmp[j] * feature->inverse_cov[j][i]);
    }
    double second = 0.;
    for (int i = 0; i < 3; i++)
        second += (first[i] * tmp[i]);
    
    /*double first = tmp[0] * (tmp[0] * features[j].inverse_cov[0][0] + tmp[1] * features[j].inverse_cov[1][0] + tmp[2] * features[j].inverse_cov[2][0]) + 
                   tmp[1] * (tmp[0] * features[j].inverse_cov[0][1] + tmp[1] * features[j].inverse_cov[1][1] + tmp[2] * features[j].inverse_cov[2][1]) +
                   tmp[2] * (tmp[0] * features[j].inverse_cov[0][2] + tmp[0] * features[j].inverse_cov[1][2] + tmp[2] * features[j].inverse_cov[2][2]);
    */
    return (-second - log(abs(feature->det_cov)));
}

void kernel(uchar4* data, int w, int h, int nc) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uchar4 p = data[y * w + x];
            double max = -1e10;
            int arg = -1;
            for (int j = 0; j < nc; j++) {
                double d = D(p, &features[j]);
                if (d > max) {
                    max = d;

                    arg = j;
                }
            }
            data[y * w + x].w = (unsigned char)arg;
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
        ull np;
        scanf("%llu", &np);
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

    memcpy(features, host_features, sizeof(AvgCov) * nc);
    for (int i = 0; i < nc; i++) {
        printf("avg = %lf %lf %lf\n", features[i].avg.x, features[i].avg.y, features[i].avg.z);
        
    }
    kernel(data, w, h, nc);



// Записываем изображение
    fp = fopen(out, "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

    free(data);
	return 0;

}