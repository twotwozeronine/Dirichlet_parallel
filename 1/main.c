#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NB 64
#define EPSILON 0.1
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef struct
{
    double (*boundary)(int, int, int, double, double);
    double (*source)(double, double);
} Problem;

// Declaration of functions
double **allocate2DArray(int N);
void initializeProblem(double **u, double **f, int N, double h, const Problem *problem);
void updateDomain(double **u, double **f, double *dm, int N, double h, int blockI, int blockJ);
int solveWavePropagation(int numBlock, double **u, double **f, double *dm, int N, double h);
void freeMemory(double **u, double **f, double *dm, int N);

// Test functions

double u1(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        return x + y;
    }
    return 0;
}

double f1(double x, double y)
{
    return 0;
}

double u2(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        return x * x + y * y;
    }
    return 0;
}

double f2(double x, double y)
{
    return -4;
}

double u3(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        return exp(x + y);
    }
    return 0;
}

double f3(double x, double y)
{
    return 2 * exp(x + y);
}

double u4(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        return sin(x) * cos(y) + x * y;
    }
    return 0;
}

double f4(double x, double y)
{
    return -sin(x) * cos(y) - cos(x) * sin(y) + 2;
}

double u5(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        return 1000.0 * pow(x, 3) * y - 2000.0 * pow(y, 4) + 500.0 * pow(y, 3) + pow(x, 2) * pow(y, 3) - 700 * x + 250 * y;
    }
    return 0;
}

double f5(double x, double y)
{
    return 6000 * x * y + 2 * pow(y, 3) + 6 * pow(x, 2) * y - 24000 * pow(y, 2) + 3000 * y;
}

double u6(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        return 10.0 * pow(x, 3) * y + 20.0 * pow(y, 3);
    }
    return 0;
}

double f6(double x, double y)
{
    return 60 * x + 120 * y;
}

double u7(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        return 100.0 * pow(x, 3) + 200.0 * pow(y, 3);
    }
    return 0;
}

double f7(double x, double y)
{
    return 600 * x + 1200 * y;
}

double u8(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        return 1000.0 * pow(x, 3) + 2000.0 * pow(y, 3);
    }
    return 0;
}

double f8(double x, double y)
{
    return 6000 * x + 12000 * y;
}

double u9(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        return 10.0 * pow(x, 4) * y + 20.0 * pow(y, 4);
    }
    return 0;
}

double f9(double x, double y)
{
    return 120 * pow(x, 2) + 240 * pow(y, 2);
}

double u10(int i, int j, int N, double x, double y)
{
    if ((i == N + 1) || (i == 0) || (j == 0) || (j == N + 1))
    {
        return 10.0 * pow(x, 5) * y + 20.0 * pow(y, 5);
    }
    return 0;
}

double f10(double x, double y)
{
    return 200 * pow(x, 3) + 400 * pow(y, 3);
}

// Array of problems
Problem problems[] = {
    {u1, f1},
    {u2, f2},
    {u3, f3},
    {u4, f4},
    {u5, f5},
    {u6, f6},
    {u7, f7},
    {u8, f8},
    {u9, f9},
    {u10, f10},
};

double **allocate2DArray(int N)
{
    double **array = (double **)malloc((N + 2) * sizeof(double *));
    for (int i = 0; i < N + 2; i++)
    {
        array[i] = (double *)malloc((N + 2) * sizeof(double));
    }
    return array;
}

void initializeProblem(double **u, double **f, int N, double h, const Problem *problem)
{
    for (int i = 0; i < N + 2; i++)
    {
        for (int j = 0; j < N + 2; j++)
        {
            double x = i * h;
            double y = j * h;
            u[i][j] = problem->boundary(i, j, N, x, y);
            f[i][j] = problem->source(x, y);
        }
    }
}

void updateDomain(double **u, double **f, double *dm, int N, double h, int blockI, int blockJ)
{
    int startI = 1 + blockI * NB;
    int endI = MIN(startI + NB, N + 1);
    int startJ = 1 + blockJ * NB;
    int endJ = MIN(startJ + NB, N + 1);
    double dm1 = 0;

    for (int i = startI; i < endI; i++)
    {
        for (int j = startJ; j < endJ; j++)
        {
            double temp = u[i][j];
            u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * f[i][j]);
            double d = fabs(temp - u[i][j]);
            if (dm1 < d)
            {
                dm1 = d;
            }
        }
    }
    if (dm[blockI] < dm1)
    {
        dm[blockI] = dm1;
    }
}

int solveWavePropagation(int numBlock, double **u, double **f, double *dm, int N, double h)
{
    int iterations = 0;
    double dmax;

    do
    {
        iterations++;
        dmax = 0.0;

        // Wave propagation: growing phase
        for (int nx = 0; nx < numBlock; nx++)
        {
            int i, j;
            dm[nx] = 0;

#pragma omp parallel for shared(nx) private(i, j)
            for (i = 0; i < nx + 1; i++)
            {
                j = nx - i;
                updateDomain(u, f, dm, N, h, j, i);
            }
        }

        // Wave propagation: damping phase
        for (int nx = numBlock - 2; nx > -1; nx--)
        {
            int i, j;

#pragma omp parallel for shared(nx) private(i, j)
            for (i = numBlock - nx - 1; i < numBlock; i++)
            {
                j = numBlock + ((numBlock - 2) - nx) - i;
                updateDomain(u, f, dm, N, h, j, i);
            }
        }

        // Determine the maximum error
        for (int i = 0; i < numBlock; i++)
        {
            if (dmax < dm[i])
            {
                dmax = dm[i];
            }
        }

    } while (dmax > EPSILON);

    return iterations;
}

void freeMemory(double **u, double **f, double *dm, int N)
{
    for (int i = 0; i < N + 2; i++)
    {
        free(u[i]);
        free(f[i]);
    }
    free(u);
    free(f);
    free(dm);
}

int main()
{
    for (int experiment = 1; experiment <= 10; experiment++)
    {
        printf("Experiment number %d.\n", experiment);
        int grids[] = {100, 300, 500, 1000, 3000};
        int numGrids = sizeof(grids) / sizeof(grids[0]);
        int threads[] = {1, 4};
        int problemIndex = experiment % (sizeof(problems) / sizeof(problems[0]));

        for (int t = 0; t < sizeof(threads) / sizeof(threads[0]); t++)
        {
            for (int n = 0; n < numGrids; n++)
            {
                double start = omp_get_wtime();
                int N = grids[n];
                double h = 1.0 / (N + 1);

                double **f = allocate2DArray(N);
                double **u = allocate2DArray(N);
                initializeProblem(u, f, N, h, &problems[problemIndex]);

                omp_set_num_threads(threads[t]);
                int numBlock = (N - 2) / NB + 1;
                double *dm = (double *)calloc(numBlock, sizeof(double));
                int iterations = solveWavePropagation(numBlock, u, f, dm, N, h);
                double end = omp_get_wtime();

                printf("N: %d, Threads: %d, Problem: %d, Time: %f sec, Iterations: %d\n", N, threads[t], problemIndex, end - start, iterations);

                freeMemory(u, f, dm, N);
            }
        }
        printf("\n");
    }

    return 0;
}