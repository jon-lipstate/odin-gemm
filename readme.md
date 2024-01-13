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

Both runs executed using: `odin test . -o:aggressive -disable-assert -microarch:native -no-bounds-check` (LLVM17)

Naive Implementation (mmult_jpi):

```
Per-Matrix Size: 0.879 mb
Clocks: 409855116, n_flops: 1769472000, time(ms):117.101, GFS:15.111
clocks/flop 0.2316256
```

Optimized version (mmult):

```
Per-Matrix Size: 0.879 mb
starting mmult
A&B Cache-Packing Temp Allocs (kb): 72 72
Clocks: 399528885, n_flops: 1769472000, time(ms):114.151, GFS:15.501
clocks/flop 0.22579
```
