//--------------------------------------------------------------
// prodAx_for_scope.c (versión paralela)
//--------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h> 
void prodAx(int m, int n, double * restrict A, double * restrict x, double * restrict b);

int main(int argc, char *argv[]) {
    double *A,*x,*b;
    int i, j, m, n;
    clock_t start, end;  // <-- Variables para medir el tiempo

    printf("Ingrese las dimensiones m y n de la matriz: ");
    scanf("%d %d",&m,&n);

    //---- Asignación de memoria para la matriz A ----
    if ( (A=(double *)malloc(m*n*sizeof(double))) == NULL )
        perror("memory allocation for A");

    //---- Asignación de memoria para el vector x ----
    if ( (x=(double *)malloc(n*sizeof(double))) == NULL )
        perror("memory allocation for x");

    //---- Asignación de memoria para el vector b ----
    if ( (b=(double *)malloc(m*sizeof(double))) == NULL )
        perror("memory allocation for b");

    printf("Initializing matrix A and vector x\n");

    //---- Inicialización con elementos aleatorios entre 1-7 y 1-13
    for (j=0; j<n; j++)
        x[j] = rand()%7+1;

    for (i=0; i<m; i++)
        for (j=0; j<n; j++)
            A[i*n+j] = rand()%13+1;

    printf("Calculando el producto Ax para m = %d n = %d\n",m,n);

    start = clock();  // <-- Iniciar medición de tiempo
    prodAx(m, n, A, x, b);
    end = clock();  // <-- Finalizar medición de tiempo

    printf("\nb: \n");
    for(j=0; j<n; j++)
        printf("\t%0.0f ",b[j]);
    printf("\n\n");

    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;  // <-- Calcular el tiempo de ejecución
    printf("Tiempo de ejecución: %f segundos\n", cpu_time_used);

    free(A);free(x);free(b);

    return(0);
}

void prodAx(int m, int n, double * restrict A, double * restrict x, double * restrict b) {
    int i, j;

    #pragma omp parallel for private(j)  // <-- Paralelizar el bucle con OpenMP
    for(i=0; i<m; i++){
        b[i]=0.0;
        for(j=0; j<n; j++){
            b[i] += A[i*n + j] * x[j];
        }
    }
}
