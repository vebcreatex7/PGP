#include <stdio.h>
#include <stdlib.h>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

// текстурная ссылка <тип элементов, размерность, режим нормализации>
texture<uchar4, 2, cudaReadModeElementType> tex;


__device__ uchar4 window(int x, int y, int width, int high, int radius) {
    short r[256] = {0};
    short g[256] = {0};
    short b[256] = {0};
    short w[256] = {0};
    
    uchar4 p;
    int x_left = max(0, x - radius);
    int x_right = min(width - 1, x + radius);
    int y_left = max(0, y - radius);
    int y_right = min(high - 1, y + radius);

    int n = (x_right - x_left + 1) * (y_right - y_left + 1);
    for (int i = x_left; i <= x_right; i++) {
        for (int j = y_left; j <= y_right; j++) {
            p = tex2D(tex, i, j);
            r[(int)p.x]++;
            g[(int)p.y]++;
            b[(int)p.z]++;
            w[(int)p.w]++;
        }
    }
    uchar4 ans = {0, 0, 0, 0};
    unsigned int s = 0;
    for (int i = 0; i < 256; i++) {
        s += r[i];
        if (s >= (n / 2) + 1) {
            ans.x = i;
            break;
        }
    }
    s = 0;

    for (int i = 0; i < 256; i++) {
        s += g[i];
        if (s >= (n / 2) + 1) {
            ans.y = i;
            break;
        }
    }
    s = 0;

    for (int i = 0; i < 256; i++) {
        s += b[i];
        if (s >= (n / 2) + 1) { 
            ans.z = i;
            break;
        }
    }
    s = 0;
    
    for (int i = 0; i < 256; i++) {
        s += w[i];
        if (s >= (n / 2) + 1) { 
            ans.w = i;
            break;
        }
    }
    return ans;
}


__global__ void kernel(uchar4 *data, int w, int h, int radius) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	for (int y = idy; y < h; y += offsety) {
        for (int x = idx; x < w; x += offsetx) {
			data[y * w + x] = window(x, y, w, h, radius);
		}
    }
}

int main() {
    char in[256];
    char out[256];
    int radius;
    scanf("%s", in);
    scanf("%s", out);
    scanf("%d", &radius);

	int w, h;
	FILE *fp = fopen(in, "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	// Подготовка данных для текстуры
	cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, w, h));

	CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

	// Подготовка текстурной ссылки, настройка интерфейса работы с данными
	tex.addressMode[0] = cudaAddressModeClamp;	// Политика обработки выхода за границы по каждому измерению
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;		// Без интерполяции при обращении по дробным координатам
	tex.normalized = false;						// Режим нормализации координат: без нормализации

	// Связываем интерфейс с данными
	CSC(cudaBindTextureToArray(tex, arr, ch));

	
    if (radius > 0) {
        uchar4 *dev_out;
	    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));
        dim3 gridSize(32,32);
        dim3 blockSize(32,32);
        kernel<<<gridSize, blockSize>>>(dev_out, w, h, radius);
	    CSC(cudaGetLastError());
	    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
        CSC(cudaFree(dev_out));
        
    }

	fp = fopen(out, "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

    // Отвязываем данные от текстурной ссылки
    CSC(cudaUnbindTexture(tex));
	CSC(cudaFreeArray(arr));
	
	free(data);
	return 0;
}
