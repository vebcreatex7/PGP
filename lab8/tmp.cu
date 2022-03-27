#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <string>
#include <iostream>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <mpi.h>

#define _i(i, j) (((j) + 1) * (nx + 2) + (i) + 1)
#define _ib(i, j) ((j) * nbx + (i))

#define BLOCKS_X 32
#define BLOCKS_Y 32
#define THREADS_X 32
#define THREADS_Y 32

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)


__global__ void kernelInit(double* data, int nx, int ny, double u0) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int j = idy; j < ny; j+=offsety) {
        for (int i = idx; i < nx; i+=offsetx) {
            data[_i(i, j)] = u0;
        }
    } 
        
}

__global__ void kernelSendLeftRight(double* buff, double* data, int ny, int nx, int xInd) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    for (int j = idx; j < ny; j+=offsetx) {
        buff[j] = data[_i(xInd, j)];
    }        
}

__global__ void kernelSendFrontBack(double* buff, double* data, int ny, int nx, int yInd) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    for (int i = idx; i < nx; i+=offsetx) {
        buff[i] = data[_i(i, yInd)];
    }        
}

__global__ void kernelReciveLeftRight(double* buff, double* data, int ny, int nx, int xInd) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    for (int j = idx; j < ny; j+=offsetx) {
        data[_i(xInd, j)] = buff[j];
    }        
}

__global__ void kernelReciveFrontBack(double* buff, double* data, int ny, int nx, int yInd) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    for (int i = idx; i < nx; i+=offsetx) {
        data[_i(i, yInd)] = buff[i];
    }        
}

__global__ void kernelSetDefaultLeftRight(double* data, int ny, int nx, int xInd, double board) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    for (int j = idx; j < ny; j+=offsetx) {
        data[_i(xInd, j)] = board;
    } 
}

__global__ void kernelSetDefaultFrontBack(double* data, int ny, int nx, int yInd, double board) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offsetx = blockDim.x * gridDim.x;

    for (int i = idx; i < nx; i+=offsetx) {
        data[_i(i, yInd)] = board;
    } 
}

__global__ void kernelStep(double* data, double *next, int ny, int nx, double hx, double hy) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int j = idy; j < ny; j+=offsety) {
        for (int i = idx; i < nx; i+=offsetx) {
            next[_i(i, j)] = 0.5 * ((data[_i(i + 1, j)] + data[_i(i - 1, j)]) / (hx * hx) +
 										(data[_i(i, j + 1)] + data[_i(i, j - 1)]) / (hy * hy)) / 
 										(1.0 / (hx * hx) + 1.0 / (hy * hy));                              
        }
    } 
}


__global__ void kernelDiff(double* data, double* next, int ny, int nx) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int j = idy - 1; j <= ny; j+=offsety) {
        for (int i = idx - 1; i <= nx; i+=offsetx) {
            data[_i(i,j)] = ((i != -1) && (j != -1) && (i != nx) && (j != ny)) *
				fabs(next[_i(i, j)] - data[_i(i, j)]);                      
        }
    } 
}

int main(int argc, char *argv[]) {
	int ib, jb, nbx, nby, nx, ny, maxn, id;

	double lx, ly, hx, hy, uFront, uBack, uLeft, uRight, u0, check, eps;
	double *data, *buff;

    char f[100];

	MPI_Status status;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	MPI_Barrier(MPI_COMM_WORLD);

    int deviceCount;
	CSC(cudaGetDeviceCount(&deviceCount));
	CSC(cudaSetDevice(id % deviceCount));

	if (id == 0) {
        std::cin >> nbx >> nby;
        std::cin >> nx >> ny;
        std::cin >> f;
        std::cin >> eps;	
        std::cin >> lx >> ly;
        std::cin >> uLeft >> uRight >> uFront >> uBack;
        std::cin >> u0;				
	}

    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nbx, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nby, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&uFront, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&uBack, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&uLeft, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&uRight, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(f, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
 
  	ib = id % nbx;
 	jb = id / nbx;

  	hx = lx / (nbx * nx);	
 	hy = ly / (nby * ny);

    maxn = fmax(nx, ny);

    double *dev_data, *dev_next, *dev_temp;

    CSC(cudaMalloc(&dev_data, sizeof(double) * (nx + 2) * (ny + 2)));
    CSC(cudaMalloc(&dev_next, sizeof(double) * (nx + 2) * (ny + 2)));

 	data = (double *)malloc(sizeof(double) * (nx + 2) * (ny + 2));
    buff = (double *)malloc(sizeof(double) * (maxn + 2));

 	int buffer_size;
 	MPI_Pack_size(maxn + 2, MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
 	buffer_size = 4 * (buffer_size + MPI_BSEND_OVERHEAD);
 	double *buffer = (double *)malloc(buffer_size);
 	MPI_Buffer_attach(buffer, buffer_size);
    
    double *dev_buff;
    CSC(cudaMalloc(&dev_buff, sizeof(double) * (maxn + 2)));


    kernelInit<<<dim3(BLOCKS_X, BLOCKS_Y), dim3(THREADS_X,THREADS_Y)>>>(dev_data, nx, ny, u0); 
 	
    check = eps + 1;
    while(check >= eps) {

        MPI_Barrier(MPI_COMM_WORLD);

 		if (ib + 1 < nbx) {
            kernelSendLeftRight<<<BLOCKS_X, THREADS_X>>>(dev_buff, dev_data, ny, nx, nx - 1);
            CSC(cudaGetLastError());

            CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * ny, cudaMemcpyDeviceToHost)); 
            MPI_Bsend(buff, ny, MPI_DOUBLE, _ib(ib + 1, jb), id, MPI_COMM_WORLD);
 		}

 		if (jb + 1 < nby) {
            kernelSendFrontBack<<<BLOCKS_X, THREADS_X>>>(dev_buff, dev_data, ny, nx, ny - 1);
            CSC(cudaGetLastError());

            CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * nx, cudaMemcpyDeviceToHost)); 
 			MPI_Bsend(buff, nx, MPI_DOUBLE, _ib(ib, jb + 1), id, MPI_COMM_WORLD);
 		}

 		if (ib > 0) {
            kernelSendLeftRight<<<BLOCKS_X, THREADS_X>>>(dev_buff, dev_data, ny, nx, 0);
            CSC(cudaGetLastError());

            CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * ny, cudaMemcpyDeviceToHost));
 			MPI_Bsend(buff, ny, MPI_DOUBLE, _ib(ib - 1, jb), id, MPI_COMM_WORLD);
 		}
		
 		if (jb > 0) {
            kernelSendFrontBack<<<BLOCKS_X, THREADS_X>>>(dev_buff, dev_data, ny, nx, 0); 
            CSC(cudaGetLastError());

            CSC(cudaMemcpy(buff, dev_buff, sizeof(double) * nx, cudaMemcpyDeviceToHost)); 
 			MPI_Bsend(buff, nx, MPI_DOUBLE, _ib(ib, jb - 1), id, MPI_COMM_WORLD);
 		}
     
     
 /*---------------------------------------------------------------------------------------------------------*/
      	if (ib > 0) {
 			MPI_Recv(buff, ny, MPI_DOUBLE, _ib(ib - 1, jb), _ib(ib - 1, jb), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * (ny + 2), cudaMemcpyHostToDevice));

            kernelReciveLeftRight<<<BLOCKS_X, THREADS_X>>>(dev_buff, dev_data, ny, nx, -1);
            CSC(cudaGetLastError());
 		} else {
            kernelSetDefaultLeftRight<<<BLOCKS_X, THREADS_X>>>(dev_data, ny, nx, -1, uLeft); 
            CSC(cudaGetLastError());
 		}

 		if (jb > 0) {
 			MPI_Recv(buff, nx, MPI_DOUBLE, _ib(ib, jb - 1), _ib(ib, jb - 1), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * (nx + 2), cudaMemcpyHostToDevice));

            kernelReciveFrontBack<<<BLOCKS_X, THREADS_X>>>(dev_buff, dev_data, ny, nx, -1); 
            CSC(cudaGetLastError());
 		} else {
            kernelSetDefaultFrontBack<<<BLOCKS_X, THREADS_X>>>(dev_data, ny, nx, -1, uFront);
            CSC(cudaGetLastError());
 		}

 		if (ib + 1 < nbx) {
 			MPI_Recv(buff, ny, MPI_DOUBLE, _ib(ib + 1, jb), _ib(ib + 1, jb), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * (ny + 2), cudaMemcpyHostToDevice));

            kernelReciveLeftRight<<<BLOCKS_X, THREADS_X>>>(dev_buff, dev_data, ny, nx, nx); 
            CSC(cudaGetLastError());
 		} else {
            kernelSetDefaultLeftRight<<<BLOCKS_X, THREADS_X>>>(dev_data, ny, nx, nx, uRight); 
            CSC(cudaGetLastError());
 		}

 		if (jb + 1 < nby) {
 			MPI_Recv(buff, nx, MPI_DOUBLE, _ib(ib, jb + 1), _ib(ib, jb + 1), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(dev_buff, buff, sizeof(double) * (nx + 2), cudaMemcpyHostToDevice));

            kernelReciveFrontBack<<<BLOCKS_X, THREADS_X>>>(dev_buff, dev_data, ny, nx, ny);
            CSC(cudaGetLastError());
 		} else {
            kernelSetDefaultFrontBack<<<BLOCKS_X, THREADS_X>>>(dev_data, ny, nx, ny, uBack);
            CSC(cudaGetLastError());
 		}
        
        MPI_Barrier(MPI_COMM_WORLD);

        kernelStep<<<dim3(BLOCKS_X, BLOCKS_Y), dim3(THREADS_X, THREADS_Y)>>>(dev_data, dev_next, ny, nx, hx, hy);
        CSC(cudaGetLastError());

        kernelDiff<<<dim3(BLOCKS_X, BLOCKS_Y), dim3(THREADS_X, THREADS_Y)>>>(dev_data, dev_next, ny, nx);
        CSC(cudaGetLastError());

        thrust::device_ptr<double> diffs = thrust::device_pointer_cast(dev_data);
        check = *thrust::max_element(diffs, diffs + (ny + 2) * (nx + 2));

		MPI_Allreduce(MPI_IN_PLACE, &check, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

       	dev_temp = dev_next;
		dev_next = dev_data;
		dev_data = dev_temp;
 	}
    
    CSC(cudaMemcpy(data, dev_data, sizeof(double) * (nx + 2) * (ny + 2), cudaMemcpyDeviceToHost));
                    
    int size = 14;
    char *buff_str = (char*) malloc(sizeof(char) * nx * ny * size);
    memset(buff_str, ' ', sizeof(char) * nx * ny * size);

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            sprintf(buff_str + (j * nx + i) * size, "%.6e", data[_i(i, j)]);
        }
    }

    for (int i = 0; i < nx * ny * size; ++i) {
        if (buff_str[i] == '\0') {
            buff_str[i] = ' ';
        }
    }

    MPI_File fp;
    MPI_Datatype filetype;

    MPI_Type_vector(ny, nx * size, nbx * nx * size, MPI_CHAR, &filetype);
	MPI_Type_commit(&filetype);

    MPI_File_delete(f, MPI_INFO_NULL);
	MPI_File_open(MPI_COMM_WORLD, f, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);

	MPI_File_set_view(fp, nbx * (id / nbx) * ny * nx * size + (id % nbx) * nx * size, MPI_CHAR, filetype, "native", MPI_INFO_NULL);
	MPI_File_write_all(fp, buff_str, nx * ny * size, MPI_CHAR, MPI_STATUS_IGNORE);

	MPI_File_close(&fp);
    MPI_Type_free(&filetype);

    free(data);
    free(buff);

    CSC(cudaFree(dev_data));
    CSC(cudaFree(dev_next));
    CSC(cudaFree(dev_buff));
    
  	MPI_Finalize();
 	return 0;
 }