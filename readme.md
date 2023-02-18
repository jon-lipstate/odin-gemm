# Odin Implementation of General Matrix Multiply (gemm)

Ported from UT-Austins BLIS project / ULAFF course. 

[Book](https://www.cs.utexas.edu/users/flame/laff/pfhp/LAFF-On-PfHP.html)

[Git](https://github.com/flame)


## Results

Odin implementation suffers substantially compared to the C-Implementation. I suspect LLVM is poorly optimizing the code, but need to run profiling and direct comparisons to get a better idea as to the root cause.

Test Computer:
```
AMD 3950X: L1: 1mb, L2: 8mb, L3:64mb (shared), GFLOPS/Core: 225, Clock: 3.5 gHz (4.7 Turbo)
Expected Performance: ~90% Max, 200 GFLOP
```

Naive Implementation (mmult_jpi):
```
Per-Matrix Size: 0.879 mb
Clocks: 8,758,405,788, n_flops: 1,769,472,000, time(ms):2502, GFS:0.707
clocks/flop 4.95
```

Optimized version (mmult):
```
Per-Matrix Size: 0.879 mb
A&B Cache-Packing Temp Allocs (kb): 72 72
Clocks: 13,652,248,133, n_flops: 1,769,472,000, time(ms):3901, GFS:0.454
clocks/flop 7.72
```