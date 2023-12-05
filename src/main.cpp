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

const INTEGER BLOCK_SIZE_1 = 64;
const INTEGER BLOCK_SIZE_2 = 1024;

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

void GEMA(INTEGER m, INTEGER n,
          double alpha, double* A, INTEGER ldA,
          double beta,  double* C, INTEGER ldC) {
    for (INTEGER j = 0; j < n; ++j) {
        for (INTEGER i = 0; i < m; ++i) {
            C[i + j * ldC] = alpha * A[i + j * ldA] + beta * C[i + j * ldC];
        }
    }
}

double *block_A;
double *block_B;
double *block_C;

void GEMM(INTEGER m, INTEGER k, INTEGER n,
               double alpha, double* A, INTEGER ldA, double* B, INTEGER ldB,
               double beta,  double* C, INTEGER ldC) {

    for (INTEGER i = 0; i < m; i += BLOCK_SIZE_1) {
        for (INTEGER j = 0; j < n; j += BLOCK_SIZE_1) {
            for (INTEGER l = 0; l < k; l += BLOCK_SIZE_1) {
                double *offset_A = A + i + l * ldA;
                double *offset_B = B + l + j * ldB;
                double *offset_C = C + i + j * ldC;

                INTEGER block_m = min(BLOCK_SIZE_1, m - i);
                INTEGER block_k = min(BLOCK_SIZE_1, k - l);
                INTEGER block_n = min(BLOCK_SIZE_1, n - j);

                for (INTEGER jj = 0; jj < block_k; ++jj) {
                    memcpy(block_A + jj * BLOCK_SIZE_1, offset_A + jj * ldA,
                            block_m);
                }
                for (INTEGER jj = 0; jj < block_n; ++jj) {
                    memcpy(block_B + jj * BLOCK_SIZE_1, offset_B + jj * ldB,
                            block_k);
                }
                for (INTEGER jj = 0; jj < block_n; ++jj) {
                    memcpy(block_C + jj * BLOCK_SIZE_1, offset_C + jj * ldC,
                            block_m);
                }

                blockGEMM(block_m, block_k, block_n, block_A, BLOCK_SIZE_1, block_B,
                        BLOCK_SIZE_1, block_C, BLOCK_SIZE_1);

                for (INTEGER jj = 0; jj < block_n; ++jj) {
                    memcpy(offset_C + jj * ldC, block_C + jj * BLOCK_SIZE_1,
                            block_m);
                }
            }
        }
    }

}

void blasDgemm(INTEGER m, INTEGER k, INTEGER n,
               double alpha, double* A, INTEGER ldA, double* B, INTEGER ldB,
               double beta,  double* C, INTEGER ldC) {
    double *AB = new double[BLOCK_SIZE_2 * BLOCK_SIZE_2];
    block_A = new double[BLOCK_SIZE_1 * BLOCK_SIZE_1];
    block_B = new double[BLOCK_SIZE_1 * BLOCK_SIZE_1];
    block_C = new double[BLOCK_SIZE_1 * BLOCK_SIZE_1];

    for (INTEGER i = 0; i < m; i += BLOCK_SIZE_2) {
        for (INTEGER j = 0; j < n; j += BLOCK_SIZE_2) {
            for (INTEGER l = 0; l < k; l += BLOCK_SIZE_2) {
                double *block_A = A + i + l * ldA;
                double *block_B = B + l + j * ldB;
                double *block_C = C + i + j * ldC;

                INTEGER block_m = min(BLOCK_SIZE_2, m - i);
                INTEGER block_k = min(BLOCK_SIZE_2, k - l);
                INTEGER block_n = min(BLOCK_SIZE_2, n - j);

                memset(AB, 0, sizeof(double) * BLOCK_SIZE_2 * BLOCK_SIZE_2);
                GEMM(block_m, block_k, block_n,
                     alpha, block_A, ldA, block_B, ldB, beta, AB, BLOCK_SIZE_2);
                GEMA(block_m, block_n, alpha, AB, BLOCK_SIZE_2, beta, C, ldC);
            }
        }
    }

    delete[] AB;
    delete[] block_A;
    delete[] block_B;
    delete[] block_C;
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
