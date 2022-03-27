#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <fstream>
#include <algorithm>
#include "mpi.h"
#include <cmath>

#define _i(i, j) (((j) + 1) * (x + 2) + (i) + 1)
#define _ibx(id) ((id) % dim1)
#define _iby(id) ((id) / dim1)
#define _ib(i, j) ((j) * dim1 + (i))

int main(int argc, char *argv[]) {
    int id, numproc, dim1, dim2, x, y, i, j, ib, jb, max;
    std::string f;
    double lx, ly, hx, hy, uLeft, uRight, uFront, uBack, u0, eps, check;
	double *data, *temp, *next, *buff;

    MPI_Status status;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Barrier(MPI_COMM_WORLD);

    if (id == 0) {
        std::cin >> dim1 >> dim2;
        std::cin >> x >> y;
        std::cin >> f;
        std::cin >> eps;
        std::cin >> lx >> ly;
        std::cin >> uLeft >> uRight >> uFront >> uBack;
        std::cin >> u0;
    }

    MPI_Bcast(&dim1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&y, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&uLeft, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&uRight, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&uFront, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&uBack, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    ib = _ibx(id);
    jb = _iby(id);
    std::cerr << ib << ' ' << jb << ' ' << _ib(ib + 1, jb) << std::endl;
    hx = lx / (double)(dim1 * x);
    hy = ly / (double)(dim2 * y);
    max = std::max(x, y);

    data = (double*)malloc(sizeof(double) * (x + 2) * (y + 2));
    next = (double*)malloc(sizeof(double) * (x + 2) * (y + 2));
    buff = (double*)malloc(sizeof(double) * (max));
    double max_check = 1000.;
    //double* check_mpi = (double*)malloc(sizeof(double) * dim1 * dim2);

    int buffer_size;
    MPI_Pack_size(max, MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
    buffer_size = 4 * (buffer_size + MPI_BSEND_OVERHEAD);
    double* buffer = (double*)malloc(buffer_size);
    MPI_Buffer_attach(buffer, buffer_size);

    for (i = 0; i < x; i++) {
        for (j = 0; j < y; j++)
            data[_i(i,j)] = u0;
    }

    for (j = 0; j < y; j++) {
        data[_i(-1,j)] = uLeft;
        next[_i(-1,j)] = uLeft;
        data[_i(x,j)] = uRight;
        next[_i(x,j)] = uRight;
    }

    for (i = 0; i < y; i++) {
        data[_i(i,-1)] = uFront;
        next[_i(i,-1)] = uFront;
        data[_i(i,y)] = uBack;
        next[_i(i,y)] = uBack;
    }



    check = eps + 1;
    while  (check >= eps){
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 1) {
            //std::cerr << check << std::endl;
        }
        if (ib + 1 < dim1) {
            for (j = 0; j < y; j++) {
                buff[j] = data[_i(x - 1, j)];
            }
                
            MPI_Bsend(buff, y, MPI_DOUBLE, _ib(ib + 1, jb), id, MPI_COMM_WORLD);
        }

        if (jb + 1 < dim2) {
            for (i = 0; i < x; i++) {
                 buff[i] = data[_i(i, y - 1)];
            }
               
            MPI_Bsend(buff, x, MPI_DOUBLE, _ib(ib, jb + 1), id, MPI_COMM_WORLD);
        }

        if (ib > 0) {
            for (j = 0; j < y; j++) {
                buff[j] = data[_i(0, j)];
            }

            MPI_Bsend(buff, y, MPI_DOUBLE, _ib(ib - 1, jb), id, MPI_COMM_WORLD);
        }

        if (jb > 0) {
            for (i = 0; i < x; i++) {
                buff[i] = data[_i(i, 0)];
            }

            MPI_Bsend(buff, x, MPI_DOUBLE, _ib(ib, jb - 1), id, MPI_COMM_WORLD);
        }

//-==========

        if (ib > 0) {
            MPI_Recv(buff, y, MPI_DOUBLE, _ib(ib - 1, jb), _ib(ib - 1, jb), MPI_COMM_WORLD, &status);
            for (j = 0; j < y; j++) {
                data[_i(-1, j)] = buff[j];
            }
        }

        if (jb > 0) {
            MPI_Recv(buff, x, MPI_DOUBLE, _ib(ib, jb - 1), _ib(ib, jb - 1), MPI_COMM_WORLD, &status);
            for (i = 0; i < x; i++) {
                data[_i(i, -1)] = buff[i];
            }
        }

        if (ib + 1 < dim1) {
            MPI_Recv(buff, y, MPI_DOUBLE, _ib(ib + 1, jb), _ib(ib + 1, jb), MPI_COMM_WORLD, &status);
            for (j = 0; j < y; j++) {
                data[_i(x, j)] = buff[j];
            }
        }

        if (jb + 1 < dim2) {
            MPI_Recv(buff, x, MPI_DOUBLE, _ib(ib, jb + 1), _ib(ib, jb + 1), MPI_COMM_WORLD, &status);
            for (i = 0; i < x; i++) {
                data[_i(i, y)] = buff[i];
            }
        }
            

        MPI_Barrier(MPI_COMM_WORLD);

        check = 0.0;
        for (i = 0; i < x; i++) {
            for (j = 0; j < y; j++) {
                next[_i(i, j)] = 0.5 * ((data[_i(i + 1, j)] + data[_i(i - 1, j)]) / (hx * hx) +
                                        (data[_i(i, j + 1)] + data[_i(i, j - 1)]) / (hy * hy)) /
                                        (1. / (hx * hx) + 1. / (hy * hy));

                check = fmax(check, fabs(next[_i(i,j)] - data[_i(i,j)]));
            }
        }

        temp = next;
        next = data;
        data = temp;

        MPI_Allreduce(MPI_IN_PLACE, &check, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }


    if (id != 0) {
        for (j = 0; j < y; j++) {
            for (i = 0; i < x; i++) {
                buff[i] = data[_i(i, j)];

                MPI_Send(buff, x, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
            }
        }
    } else {
        FILE* output = fopen(f.c_str(), "w");

        for (jb = 0; jb < dim2; jb++) {
            for (int j = 0; j < y; j++) {
                for (ib = 0; ib < dim1; ib++) {
                    if (_ib(ib, jb) == 0) {
                        for (i = 0; i < x; i++) {
                            buff[i] = data[_i(i, j)];
                        }
                    } else {
                        MPI_Recv(buff, x, MPI_DOUBLE, _ib(ib, jb), _ib(ib, jb), MPI_COMM_WORLD, &status);
                    }

                    for (i = 0; i < x; i++) {
                        fprintf(output, "%f ", buff[i]);
                        //file << buff[i] << " ";
                    }

                    if (ib + 1 == dim1) {
                        fprintf(output, "\n");
                        //file << "\n";
                        if (j == y)
                            fprintf(output, "\n");
                            //file << "\n";
                    } else {
                        fprintf(output, " ");
                        //file << ' ';
                    }
                }
            }
        }
        fclose(output);
        //file.close();
    }
    MPI_Finalize();

    free(data);
    free(next);
    free(buff);

    return 0;
}