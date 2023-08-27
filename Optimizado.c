#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void prodAx_optimized(int m, int n, double * restrict A, double * restrict x, double * restrict b);

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

    start = clock();
    prodAx_optimized(m, n, A, x, b);
    end = clock();

    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Tiempo de ejecución: %f segundos\n", cpu_time_used);

    free(A);
    free(x);
    free(b);

    return(0);
}

void prodAx_optimized(int m, int n, double * restrict A, double * restrict x, double * restrict b) {
    int i, j;

    #pragma omp parallel for private(j) schedule(guided, 1000) reduction(+:b[:m])
    for(i=0; i<m; i++) {
        b[i] = 0.0;
        for(j=0; j<n; j++) {
            b[i] += A[i*n + j] * x[j];
        }
    }
}
