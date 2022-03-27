#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <string>
#include "mpi.h"

#define _i(i, j) (((j) + 1) * (nx + 2) + (i) + 1)
#define _ib(i, j) ((j) * nbx + (i))

int main(int argc, char *argv[]) {
	int ib, jb, nbx, nby, nx, ny, maxn;
	int i, j, id;

	double lx, ly, hx, hy, uFront, uBack, uLeft, uRight, u0, check, eps;
	double *data, *temp, *next, *buff;

	std::string f;

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

	ib = id % nbx;
	jb = id / nbx;

	hx = lx / (nx * nbx);	
	hy = ly / (ny * nby);

	maxn = fmax(nx, ny);

	data = (double *)malloc(sizeof(double) * (nx + 2) * (ny + 2));	
	next = (double *)malloc(sizeof(double) * (nx + 2) * (ny + 2));
	buff = (double *)malloc(sizeof(double) * (maxn + 2));

	int buffer_size;
	MPI_Pack_size(maxn + 2, MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
	buffer_size = 4 * (buffer_size + MPI_BSEND_OVERHEAD);
	double *buffer = (double *)malloc(buffer_size);
	MPI_Buffer_attach(buffer, buffer_size);

	for(i = 0; i < nx; i++)
		for(j = 0; j < ny; j++)
			data[_i(i, j)] = u0;

	check = eps + 1;
	while(check >= eps) {
		MPI_Barrier(MPI_COMM_WORLD);

		if (ib + 1 < nbx) {					
			for(j = 0; j < ny; j++)
				buff[j] = data[_i(nx - 1, j)];
			MPI_Bsend(buff, ny, MPI_DOUBLE, _ib(ib + 1, jb), id, MPI_COMM_WORLD);
		}

		if (jb + 1 < nby) {
			for(i = 0; i < nx; i++)
				buff[i] = data[_i(i, ny - 1)];
			MPI_Bsend(buff, nx, MPI_DOUBLE, _ib(ib, jb + 1), id, MPI_COMM_WORLD);
		}

		if (ib > 0) {
			for(j = 0; j < ny; j++)
				buff[j] = data[_i(0, j)];
			MPI_Bsend(buff, ny, MPI_DOUBLE, _ib(ib - 1, jb), id, MPI_COMM_WORLD);
		}
		
		if (jb > 0) {
			for(i = 0; i < nx; i++)
				buff[i] = data[_i(i, 0)];
			MPI_Bsend(buff, nx, MPI_DOUBLE, _ib(ib, jb - 1), id, MPI_COMM_WORLD);
		}
/*---------------------------------------------------------------------------------------------------------*/
     	if (ib > 0) {
			MPI_Recv(buff, ny, MPI_DOUBLE, _ib(ib - 1, jb), _ib(ib - 1, jb), MPI_COMM_WORLD, &status);
			for(j = 0; j < ny; j++)
				data[_i(-1, j)] = buff[j];
		} else {
			for(j = 0; j < ny; j++)
				data[_i(-1, j)] = uLeft;
		}

		if (jb > 0) {
			MPI_Recv(buff, nx, MPI_DOUBLE, _ib(ib, jb - 1), _ib(ib, jb - 1), MPI_COMM_WORLD, &status);
			for(i = 0; i < nx; i++)
				data[_i(i, -1)] = buff[i];
		} else {
			for(i = 0; i < nx; i++)
				data[_i(i, -1)] = uFront;
		}

		if (ib + 1 < nbx) {
			MPI_Recv(buff, ny, MPI_DOUBLE, _ib(ib + 1, jb), _ib(ib + 1, jb), MPI_COMM_WORLD, &status);
			for(j = 0; j < ny; j++)
				data[_i(nx, j)] = buff[j];
		} else {
			for(j = 0; j < ny; j++)
				data[_i(nx, j)] = uRight;
		}

		if (jb + 1 < nby) {
			MPI_Recv(buff, nx, MPI_DOUBLE, _ib(ib, jb + 1), _ib(ib, jb + 1), MPI_COMM_WORLD, &status);
			for(i = 0; i < nx; i++)
				data[_i(i, ny)] = buff[i];
		} else {
			for(i = 0; i < nx; i++)
				data[_i(i, ny)] = uBack;
		}

		MPI_Barrier(MPI_COMM_WORLD);

		check = 0.;
		for(i = 0; i < nx; i++)
			for(j = 0; j < ny; j++) {
				next[_i(i, j)] = 0.5 * ((data[_i(i + 1, j)] + data[_i(i - 1, j)]) / (hx * hx) +
										(data[_i(i, j + 1)] + data[_i(i, j - 1)]) / (hy * hy)) / 
											(1.0 / (hx * hx) + 1.0 / (hy * hy));

				check = fmax(check, fabs(next[_i(i,j)] - data[_i(i,j)]));
			}
				
		temp = next;
		next = data;
		data = temp;

		MPI_Allreduce(MPI_IN_PLACE, &check, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	}

	if (id != 0) {	
		for(j = 0; j < ny; j++) {
			for(i = 0; i < nx; i++) 
				buff[i] = data[_i(i, j)];
			MPI_Send(buff, nx, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
		}
	} else {
		FILE* output = fopen(f.c_str(), "w");

		for(jb = 0; jb < nby; jb++)
			for(j = 0; j < ny; j++)
				for(ib = 0; ib < nbx; ib++) {
					if (_ib(ib, jb) == 0)
						for(i = 0; i < nx; i++)
							buff[i] = data[_i(i, j)];
					else
						MPI_Recv(buff, nx, MPI_DOUBLE, _ib(ib, jb), _ib(ib, jb), MPI_COMM_WORLD, &status);
					for(i = 0; i < nx; i++)
						fprintf(output,"%f ", buff[i]);
					
					if (ib + 1 == nbx) {
                        fprintf(output, "\n");
                        //file << "\n";
                        if (j == ny)
                            fprintf(output, "\n");
                            //file << "\n";
                    } else {
                        fprintf(output, " ");
                        //file << ' ';
                    }
				}
		fclose(output);
	}

	MPI_Finalize();

    free(data);
    free(next);
    free(buff);

    return 0;
}