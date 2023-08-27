#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void prodAx(int m, int n, double * restrict A, double * restrict x, double * restrict b, char* schedule, int block_size);

int main(int argc, char *argv[]) {
    double *A,*x,*b;
    int i, j;
    clock_t start, end;

    int m = 30000;
    int n = 30000;

    printf("Dimensiones de la matriz: %d x %d\n", m, n);

    A = (double *)malloc(m*n*sizeof(double));
    x = (double *)malloc(n*sizeof(double));
    b = (double *)malloc(m*sizeof(double));

    printf("Initializing matrix A and vector x\n");

    for (j=0; j<n; j++)
        x[j] = rand()%7+1;

    for (i=0; i<m; i++)
        for (j=0; j<n; j++)
            A[i*n+j] = rand()%13+1;

    char* schedules[] = {"static", "dynamic", "guided"};
    int static_blocks[] = {100000, 10000, 1000};
    int dynamic_blocks[] = {100000, 10000, 1000};
    int guided_blocks[] = {1000, 100, 10};
    int* blocks[] = {static_blocks, dynamic_blocks, guided_blocks};
    int block_counts[] = {3, 3, 3};

    for (int s = 0; s < 3; s++) {
        for (int block_idx = 0; block_idx < block_counts[s]; block_idx++) {
            start = clock();
            prodAx(m, n, A, x, b, schedules[s], blocks[s][block_idx]);
            end = clock();

            double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
            printf("Tiempo de ejecución con planificación %s y tamaño de bloque %d: %f segundos\n", schedules[s], blocks[s][block_idx], cpu_time_used);
        }
    }

    free(A);
    free(x);
    free(b);

    return(0);
}

void prodAx(int m, int n, double * restrict A, double * restrict x, double * restrict b, char* schedule, int block_size) {
    int i, j;

    if (strcmp(schedule, "static") == 0) {
        #pragma omp parallel for private(j) schedule(static, block_size)
        for(i=0; i<m; i++) {
            b[i]=0.0;
            for(j=0; j<n; j++){
                b[i] += A[i*n + j] * x[j];
            }
        }
    } else if (strcmp(schedule, "dynamic") == 0) {
        #pragma omp parallel for private(j) schedule(dynamic, block_size)
        for(i=0; i<m; i++) {
            b[i]=0.0;
            for(j=0; j<n; j++){
                b[i] += A[i*n + j] * x[j];
            }
        }
    } else if (strcmp(schedule, "guided") == 0) {
        #pragma omp parallel for private(j) schedule(guided, block_size)
        for(i=0; i<m; i++) {
            b[i]=0.0;
            for(j=0; j<n; j++){
                b[i] += A[i*n + j] * x[j];
            }
        }
    }
}
