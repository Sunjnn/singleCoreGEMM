#include <vector>
#include <memory>
#include <iostream>
#include <chrono>

#include <cstdlib>

#include <string.h>

using std::vector;
using std::rand;
using std::cout, std::endl;
using INTEGER = size_t;
using std::chrono::system_clock, std::chrono::duration_cast;
using std::chrono::microseconds;

template<class T>
void initMatrix(T* matrix, size_t m, size_t n) {
    for (int i = 0; i < m * n; ++i) matrix[i] = 1.0;
}

template<class T>
inline T min(T a, T b) {
    if (a < b) return a;
    return b;
}

void blockGEMM(INTEGER m, INTEGER k, INTEGER n,
               double alpha, double* A, INTEGER ldA, double* B, INTEGER ldB,
               double beta,  double* C, INTEGER ldC) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double inner_prod = 0.0;
            for (size_t l = 0; l < k; ++l) {
                inner_prod += A[i + l * ldA] * B[l + j * ldB];
            }
            C[i + j * ldC] = alpha * inner_prod + beta * C[i + j * ldC];
        }
    }
}

void GEMM(INTEGER m, INTEGER k, INTEGER n, double alpha, double* A, INTEGER ldA,
          double* B, INTEGER ldB, double beta, double* C, INTEGER ldC) {
    INTEGER block_m = 108;
    INTEGER block_k = 108;
    INTEGER block_n = 108;

    INTEGER num_m = (m + block_m - 1) / block_m;
    INTEGER num_k = (k + block_k - 1) / block_k;
    INTEGER num_n = (n + block_n - 1) / block_n;

    for (INTEGER block_idm = 0; block_idm < num_m; ++block_idm) {
        for (INTEGER block_idn = 0; block_idn < num_n; ++block_idn) {
            double *block_A = A + block_idm * block_m;
            double *block_B = B + block_idn * block_n * ldB;
            double *block_C = C + block_idn * block_n * ldC
                                + block_idm * block_m;

            for (INTEGER block_idk = 0; block_idk < num_k; ++block_idk) {
                INTEGER cur_m = min(block_m, m - block_m * block_idm);
                INTEGER cur_k = min(block_k, k - block_k * block_idk);
                INTEGER cur_n = min(block_n, n - block_n * block_idn);
                blockGEMM(cur_m, cur_k, cur_n,
                          alpha, block_A, ldA, block_B, ldB,
                          beta,  block_C, ldC);
                block_A += block_k * ldA;
                block_B += block_k;
            }
        }
    }
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

    GEMM(size, size, size, 1.0, A, size, B, size, 0.0, C, size);

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
