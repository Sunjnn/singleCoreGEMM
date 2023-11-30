#include <vector>
#include <memory>
#include <iostream>
#include <chrono>

#include <cstdlib>

#include <string.h>

#include <immintrin.h>

using std::vector;
using std::rand;
using std::cout, std::endl;
using INTEGER = size_t;
using std::chrono::system_clock, std::chrono::duration_cast;
using std::chrono::microseconds;

const INTEGER BLOCK_SIZE = 3072;

template<class T>
void initMatrix(T* matrix, size_t m, size_t n) {
    for (int i = 0; i < m * n; ++i) matrix[i] = 1.0;
}

template<class T>
inline T min(T a, T b) {
    if (a < b) return a;
    return b;
}

template<class T>
inline T divUP(T a, T b) {
    return (a + b - 1) / b;
}

inline void blockGEMM(INTEGER m, INTEGER k, INTEGER n,
               double* A, INTEGER ldA, double* B, INTEGER ldB,
               double* C, INTEGER ldC) {
    __m512d ra, rb, rc;
    for (size_t j = 0; j < n; ++j) {
        for (size_t l = 0; l < k; ++l) {
            rb = _mm512_set1_pd(B[l + j * ldB]);
            for (size_t i = 0; i < m; i += 8) {
                if (m - i < 8) {
                    while (i < m) {
                        C[i + j * ldC] += A[i + l * ldA] * B[l + j * ldB];
                        ++i;
                    }
                    break;
                }

                ra = _mm512_loadu_pd(A + i + l * ldA);
                rc = _mm512_loadu_pd(C + i + j * ldC);

                rc = _mm512_fmadd_pd(ra, rb, rc);

                _mm512_storeu_pd(C + i + j * ldC, rc);
            }
        }
    }
}

template<class T>
void div2Matrix(T* matrix,
                T*& matrix11, T*& matrix12, T*& matrix21, T*& matrix22,
                INTEGER m, INTEGER n, INTEGER ld) {
    INTEGER block_m = divUP(m, (INTEGER)2);
    INTEGER block_n = divUP(n, (INTEGER)2);

    matrix11 = matrix;
    matrix21 = matrix + block_m;
    matrix12 = matrix + ld * block_n;
    matrix22 = matrix12 + block_m;
}

void GEMM(INTEGER m, INTEGER k, INTEGER n,
          double* A, INTEGER ldA, double* B, INTEGER ldB,
          double* C, INTEGER ldC) {
    if (m * k + k * n + m * n <= BLOCK_SIZE) {
        blockGEMM(m, k, n, A, ldA, B, ldB, C, ldC);
        return;
    }

    double *A11, *A12, *A21, *A22;
    div2Matrix(A, A11, A12, A21, A22, m, k, ldA);
    double *B11, *B12, *B21, *B22;
    div2Matrix(B, B11, B12, B21, B22, m, k, ldB);
    double *C11, *C12, *C21, *C22;
    div2Matrix(C, C11, C12, C21, C22, m, k, ldC);

    INTEGER block_m = divUP(m, (INTEGER)2);
    INTEGER block_k = divUP(k, (INTEGER)2);
    INTEGER block_n = divUP(n, (INTEGER)2);

    GEMM(block_m, block_k, block_n, A11, ldA, B11, ldB, C11, ldC);
    GEMM(block_m, k - block_k, block_n, A12, ldA, B21, ldB, C11, ldC);

    GEMM(block_m, block_k, n - block_n, A11, ldA, B12, ldB, C12, ldC);
    GEMM(block_m, k - block_k, n - block_n, A12, ldA, B22, ldB, C12, ldC);

    GEMM(m - block_m, block_k, block_n, A21, ldA, B11, ldB, C21, ldC);
    GEMM(m - block_m, k - block_k, block_n, A22, ldA, B21, ldB, C21, ldC);

    GEMM(m - block_m, block_k, n - block_n, A21, ldA, B12, ldB, C22, ldC);
    GEMM(m - block_m, k - block_k, n - block_n, A22, ldA, B22, ldB, C22, ldC);
}

void GEMA(INTEGER m, INTEGER n,
          double alpha, double* A, INTEGER ldA,
          double beta,  double* C, INTEGER ldC) {
    for (INTEGER j = 0; j < n; ++j) {
        for (INTEGER i = 0; i < m; ++i) {
            C[i + j * ldC] = alpha * A[i + j * ldA] + beta * C[i + j * ldC];
        }
    }
}

void blasDgemm(INTEGER m, INTEGER k, INTEGER n,
               double alpha, double* A, INTEGER ldA, double* B, INTEGER ldB,
               double beta,  double* C, INTEGER ldC) {
    if (m * n + m * k + n * k <= BLOCK_SIZE) {
        double *AB = new double[m * n];
        memset(AB, 0, sizeof(double) * m * n);

        blockGEMM(m, k, n, A, ldA, B, ldB, AB, m);
        GEMA(m, n, alpha, AB, m, beta, C, ldC);

        delete[] AB;
        return;
    }

    INTEGER block_m = divUP(m, (INTEGER)2);
    INTEGER block_k = divUP(k, (INTEGER)2);
    INTEGER block_n = divUP(n, (INTEGER)2);

    double *AB = new double[block_m * block_n];

    double *A11, *A12, *A21, *A22;
    div2Matrix(A, A11, A12, A21, A22, m, k, ldA);
    double *B11, *B12, *B21, *B22;
    div2Matrix(B, B11, B12, B21, B22, m, k, ldB);
    double *C11, *C12, *C21, *C22;
    div2Matrix(C, C11, C12, C21, C22, m, k, ldC);

    memset(AB, 0, sizeof(double) * block_m * block_n);
    GEMM(block_m, block_k, block_n, A11, ldA, B11, ldB, AB, block_m);
    GEMM(block_m, k - block_k, block_n, A12, ldA, B21, ldB, AB, block_m);
    GEMA(block_m, block_n, alpha, AB, block_m, beta, C11, ldC);

    memset(AB, 0, sizeof(double) * block_m * block_n);
    GEMM(block_m, block_k, n - block_n, A11, ldA, B21, ldB, AB, block_m);
    GEMM(block_m, k - block_k, n - block_n, A12, ldA, B22, ldB, AB, block_m);
    GEMA(block_m, n - block_n, alpha, AB, block_m, beta, C12, ldC);

    memset(AB, 0, sizeof(double) * block_m * block_n);
    GEMM(m - block_m, block_k, block_n, A21, ldA, B11, ldB, AB, block_m);
    GEMM(m - block_m, k - block_k, block_n, A22, ldA, B21, ldB, AB, block_m);
    GEMA(m - block_m, block_n, alpha, AB, block_m, beta, C21, ldC);

    memset(AB, 0, sizeof(double) * block_m * block_n);
    GEMM(m - block_m, block_k, n - block_n, A21, ldA, B12, ldB, AB, block_m);
    GEMM(m - block_m, k - block_k, n - block_n, A22, ldA, B22, ldB, AB, block_m);
    GEMA(block_m, block_n, alpha, AB, block_m, beta, C22, ldC);

    delete[] AB;
}

template<class CHAR, class INT>
INT convertChartoInt(CHAR* str, INT size) {
    INT res = 0;
    size_t len = strlen(str);

    for (int i = 0; i < len; ++i) {
        res = res * 10 + (str[i] - '0');
    }
    return res;
}

int main(int argc, char** argv)
{
    INTEGER size = 8192;
    if (argc > 1) {
        size = convertChartoInt(argv[1], size);
        cout << "Set size of matrix to " << size << endl;
    }

    cout << "---------------------------------------------------------" <<endl;
    cout << "Test performance of GEMM" << endl;
    auto start = system_clock::now();

    double *A = new double[size * size];
    double *B = new double[size * size];
    double *C = new double[size * size];

    initMatrix(A, size, size);
    initMatrix(B, size, size);

    blasDgemm(size, size, size, 1.0, A, size, B, size, 0.0, C, size);

    delete[] A;
    delete[] B;
    delete[] C;

    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "openblas with size " << size << " time (us): ";
    cout << duration.count() << endl;

    double num_flop = 2.0 * size * size * size;
    double flops = num_flop / duration.count() * 1e6;
    cout << "openblas with size " << size << " performance (FLOPS): ";
    cout << flops << endl << endl << endl;

    return 0;
}
