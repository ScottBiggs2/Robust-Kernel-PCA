A python implementation of 'Exactly Robust kernel PCA'

GitHub: https://github.com/jicongfan/RKPCA_TNNLS2019/blob/master/RKPCA_PLMAdSS.m 

Paper: https://arxiv.org/abs/1802.10558

Expected return from main.py: 

```
Generating synthetic nonlinear data...
Data shape: (7690, 10)
True rank of clean data: 6
Noise density: 30.06%

==================================================
EXAMPLE 1: Standard RKPCA Decomposition
==================================================
Estimated sigma: 115.6651
lambda=0.000079
iteration=1/50  obj=7.798667  stopC=1.541022  tau=0.000012  c=0.100000

Recovery Results:
Clean data recovery error: 2.2941
Noise recovery error: 4.0180
Reconstruction error: 0.000000

==================================================
EXAMPLE 2: RKPCA with Data Compression
==================================================

--- Compressing to 8 components ---
Original shape: (7690, 10)
Compressed shape: (8, 10)
Compression ratio: 961.25x
Explained variance: 0.8958
Reconstruction error: 1.6156

--- Compressing to 16 components ---
Original shape: (7690, 10)
Compressed shape: (9, 10)
Compression ratio: 480.62x
Explained variance: 1.0000
Reconstruction error: 1.6156

--- Compressing to 32 components ---
Original shape: (7690, 10)
Compressed shape: (9, 10)
Compression ratio: 240.31x
Explained variance: 1.0000
Reconstruction error: 1.6156

--- Compressing to 64 components ---
Original shape: (7690, 10)
Compressed shape: (9, 10)
Compression ratio: 120.16x
Explained variance: 1.0000
Reconstruction error: 1.6156

--- Compressing to 128 components ---
Original shape: (7690, 10)
Compressed shape: (9, 10)
Compression ratio: 60.08x
Explained variance: 1.0000
Reconstruction error: 1.6156

--- Compressing to 256 components ---
Original shape: (7690, 10)
Compressed shape: (9, 10)
Compression ratio: 30.04x
Explained variance: 1.0000
Reconstruction error: 1.6156

==================================================
EXAMPLE 3: Comparison with sklearn KernelPCA
==================================================
Sklearn KernelPCA (clean data) shape: (5, 10)
Robust KPCA (noisy data) shape: (5, 10)
Sklearn explained variance: 1.0000
Robust KPCA explained variance: 0.5688

==================================================
EXAMPLE 4: Transform New Data
==================================================
Test data original shape: (7690, 10)
Test data compressed shape: (5, 10)

==================================================
Convergence Plot
==================================================
```
