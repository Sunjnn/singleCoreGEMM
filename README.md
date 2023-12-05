# singleCoreGEMM

[Homework 1](https://sites.google.com/lbl.gov/cs267-spr2022/hw-1) of cs267.

Implement GEMM by 2 level of blocking.
First level (function `blasDgemm`) execute matrix multiplication $A \times B$
which are tiled to size [1024, 1024] and second level (function `GEMM`) execute
multiplication which are tiled to size [64, 64]. Function `GEMM` repacks input
matrices. Function `blockGEMM` execute matrix multiplication with size [64, 64].
Avx512 are used to execute fused mul-add instruction.

This route achieves half performance of MKL.

## build and run

### manually

Build this project using `xmake`.
Follow commands below.

``` bash
xmake
```

Run using `xmake`.
Follow commands below.

``` bash
xmake run singleCoreGEMM_BMOpenBLAS [size]
```

### automatically

Run the script `run.sh` to build and run automatically.
It will execute GEMM with sizes [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768].

``` bash
bash run.sh
```
