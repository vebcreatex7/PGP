#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <string>
#include <iostream>
#include <cstring>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define _i(i, j) (((j) + 1) * (nx + 2) + (i) + 1)
#define _ib(i, j) ((j) * nbx + (i))

int main(int argc, char *argv[]) {
	int ib, jb, nbx, nby, nx, ny, maxn;
	int i, j, id;

	double lx, ly, hx, hy, uFront, uBack, uLeft, uRight, u0, check, eps;
	double *data, *temp, *next;

	char f[100];

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	MPI_Barrier(MPI_COMM_WORLD);

	if (id == 0) {
		std::cin >> nbx >> nby;
		std::cin >> nx >> ny;
		std::cin >> f;
		std::cin >> eps;
		std::cin >> lx >> ly;
		std::cin >> uLeft >> uRight >> uFront >> uBack;
		std::cin >> u0;
	}

	MPI_Bcast(&nbx, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nby, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
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

	hx = lx / (nx * nbx);	
	hy = ly / (ny * nby);

	maxn = fmax(nx, ny);

	data = (double *)malloc(sizeof(double) * (nx + 2) * (ny + 2));	
	next = (double *)malloc(sizeof(double) * (nx + 2) * (ny + 2));

	int buffer_size;
	MPI_Pack_size(maxn + 2, MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
	buffer_size = 4 * (buffer_size + MPI_BSEND_OVERHEAD);
	double *buffer = (double *)malloc(buffer_size);
	MPI_Buffer_attach(buffer, buffer_size);

	MPI_Datatype rightSend;
    MPI_Type_vector(ny, 1, nx + 2, MPI_DOUBLE, &rightSend);
	MPI_Type_commit(&rightSend);

    MPI_Datatype backSend;
    MPI_Type_vector(nx, 1, 1, MPI_DOUBLE, &backSend);
    MPI_Type_commit(&backSend);

    MPI_Datatype leftSend;
    MPI_Type_vector(ny, 1, nx + 2, MPI_DOUBLE, &leftSend);
    MPI_Type_commit(&leftSend);

    MPI_Datatype frontSend;
    MPI_Type_vector(nx, 1, 1, MPI_DOUBLE, &frontSend);
    MPI_Type_commit(&frontSend);

    MPI_Datatype leftRecv;
    MPI_Type_vector(ny, 1, nx + 2, MPI_DOUBLE, &leftRecv);
    MPI_Type_commit(&leftRecv);

    MPI_Datatype frontRecv;
    MPI_Type_vector(nx, 1, 1, MPI_DOUBLE, &frontRecv);
    MPI_Type_commit(&frontRecv);

    MPI_Datatype rightRecv;
    MPI_Type_vector(ny, 1, nx + 2, MPI_DOUBLE, &rightRecv);
	MPI_Type_commit(&rightRecv);

    MPI_Datatype backRecv;
    MPI_Type_vector(nx, 1, 1, MPI_DOUBLE, &backRecv);
    MPI_Type_commit(&backRecv);

	#pragma omp parallel for private(i,j) shared(data, u0)
	for(i = 0; i < nx; i++)
		for(j = 0; j < ny; j++)
			data[_i(i, j)] = u0;

	check = eps + 1;
	while(check >= eps) {
		MPI_Barrier(MPI_COMM_WORLD);

		if (ib + 1 < nbx) {					
			MPI_Bsend(data + nx + 2 + nx, 1, rightSend, _ib(ib + 1, jb), id, MPI_COMM_WORLD);
		}

		if (jb + 1 < nby) {
			MPI_Bsend(data + (nx + 2) * ny + 1, 1, backSend, _ib(ib, jb + 1), id, MPI_COMM_WORLD);
		}

		if (ib > 0) {
			MPI_Bsend(data + nx + 2 + 1, 1, leftSend, _ib(ib - 1, jb), id, MPI_COMM_WORLD);
 		}
		
		if (jb > 0) {
			MPI_Bsend(data + nx + 2 + 1, 1, frontSend, _ib(ib, jb - 1), id, MPI_COMM_WORLD);
		}
/*---------------------------------------------------------------------------------------------------------*/
     	if (ib > 0) {
			MPI_Recv(data + nx + 2, 1, leftRecv, _ib(ib - 1, jb), _ib(ib - 1, jb), MPI_COMM_WORLD, &status);
		} else {
			for(j = 0; j < ny; j++)
				data[_i(-1, j)] = uLeft;
		}

		if (jb > 0) {
			MPI_Recv(data + 1, 1, frontRecv, _ib(ib, jb - 1), _ib(ib, jb - 1), MPI_COMM_WORLD, &status);
		} else {
			for(i = 0; i < nx; i++)
				data[_i(i, -1)] = uFront;
		}

		if (ib + 1 < nbx) {
			MPI_Recv(data + nx + 2 + nx + 1, 1, rightRecv, _ib(ib + 1, jb), _ib(ib + 1, jb), MPI_COMM_WORLD, &status);
		} else {
			for(j = 0; j < ny; j++)
				data[_i(nx, j)] = uRight;
		}

		if (jb + 1 < nby) {
			MPI_Recv(data + (ny + 1) * (nx + 2) + 1, 1, backRecv, _ib(ib, jb + 1), _ib(ib, jb + 1), MPI_COMM_WORLD, &status);
		} else {
			for(i = 0; i < nx; i++)
				data[_i(i, ny)] = uBack;
		}

		MPI_Barrier(MPI_COMM_WORLD);

		check = 0.;

		#pragma omp parallel for private(i,j) shared(data, next,nx,ny,hx,hy) reduction(max:check)
		for(i = 0; i < nx; i++) {
			for(j = 0; j < ny; j++) {
				next[_i(i, j)] = 0.5 * ((data[_i(i + 1, j)] + data[_i(i - 1, j)]) / (hx * hx) +
										(data[_i(i, j + 1)] + data[_i(i, j - 1)]) / (hy * hy)) / 
											(1.0 / (hx * hx) + 1.0 / (hy * hy));

				check = std::max(check, fabs(next[_i(i,j)] - data[_i(i,j)]));
			}
		}	

		MPI_Allreduce(MPI_IN_PLACE, &check, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		temp = next;
		next = data;
		data = temp;
	}

	int size = 14;
    char *buff = (char*) malloc(sizeof(char) * nx * ny * size);
    memset(buff, ' ', sizeof(char) * nx * ny * size);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            sprintf(buff + (j * nx + i) * size, "%.6e", data[_i(i, j)]);
        }
    }

    for (int i = 0; i < nx * ny * size; ++i) {
        if (buff[i] == '\0')
            buff[i] = ' ';    
    }

    MPI_File fp;
    MPI_Datatype filetype;

    MPI_Type_vector(ny, nx * size, nbx * nx * size, MPI_CHAR, &filetype);
	MPI_Type_commit(&filetype);

    MPI_File_delete(f, MPI_INFO_NULL);
	MPI_File_open(MPI_COMM_WORLD, f, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
	MPI_File_set_view(fp, nbx * (id / nbx) * ny * nx * size + (id % nbx) * nx * size, MPI_CHAR, filetype, "native", MPI_INFO_NULL);
	MPI_File_write_all(fp, buff, nx * ny * size, MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&fp);
    MPI_Type_free(&filetype);

	MPI_Finalize();

    free(data);
    free(next);

    return 0;
}