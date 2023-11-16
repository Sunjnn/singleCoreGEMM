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

void GEMM(INTEGER m, INTEGER k, INTEGER n, double alpha, double* A, INTEGER ldA,
          double* B, INTEGER ldB, double beta, double* C, INTEGER ldC) {
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
