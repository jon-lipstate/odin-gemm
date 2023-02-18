package gemm
import "core:mem"
import "core:simd"
import "core:testing"

main :: proc() {
	perf_test(nil)
}
// Naive Implementation
mmult_jpi :: proc(A: ^Matrix, B: ^Matrix, C: ^Matrix) #no_bounds_check {
	k := int(A.n_rows)
	m := int(A.n_cols)
	n := int(B.n_cols)
	for j := 0; j < n; j += 1 {
		for p := 0; p < k; p += 1 {
			for i := 0; i < m; i += 1 {
				get(C, i, j)^ += get(A, i, p)^ * get(B, p, j)^
			}
		}
	}
}

Matrix :: struct {
	// offset: [2]int,
	n_rows: int,
	n_cols: int,
	base:   [^]f64,
}
// Col-Major
Matrix_Slice :: struct {
	base:   [^]f64,
	stride: int,
}
submatrix :: proc(mat: ^Matrix, r, c: int) -> Matrix {
	m := Matrix{}
	// m.offset = mat.offset + {r, c}
	m.n_rows = mat.n_rows
	m.n_cols = mat.n_cols
	m.base = mem.ptr_offset(mat.base, int(m.n_cols) * r + c)
	return m
}

get :: proc {
	pget,
	mget,
	sget,
}
pget :: #force_inline proc(ptr: [^]f64, cstride, r, c: int) -> ^f64 {
	result := &ptr[c * cstride + r]
	return result
}
mget :: #force_inline proc(mat: ^Matrix, r, c: int) -> ^f64 {
	// i is rows
	// j is cols
	result := &mat.base[c * mat.n_cols + r]
	return result
}
sget :: #force_inline proc(s: ^Matrix_Slice, r, c: int) -> ^f64 {
	result := &s.base[c * s.stride + r]
	return result
}
get_slice :: proc {
	get_slice_ms,
	get_slice_m,
}
// Strides are passed down untouched
get_slice_ms :: #force_inline proc "contextless" (v: ^Matrix_Slice, r, c: int) -> Matrix_Slice {
	base := &v.base[r + c * v.stride]
	result := Matrix_Slice{base, v.stride}
	return result
}
get_slice_m :: #force_inline proc "contextless" (v: ^Matrix, r, c: int) -> Matrix_Slice {
	base := &v.base[c * v.n_rows + r]
	result := Matrix_Slice{base, v.n_rows}
	return result
}
set_random :: proc(mat: ^Matrix) #no_bounds_check {
	r := rand.create(1)
	for i: int = 0; i < mat.n_rows; i += 1 {
		for j: int = 0; j < mat.n_cols; j += 1 {
			value := rand.float64_range(-10, 10, &r)
			get(mat, i, j)^ = value
		}
	}
}
set_ones :: proc(mat: ^Matrix) #no_bounds_check {
	for i: int = 0; i < mat.n_rows; i += 1 {
		for j: int = 0; j < mat.n_cols; j += 1 {
			mget(mat, i, j)^ = 1
		}
	}
}
set_n :: proc(mat: ^Matrix) #no_bounds_check {
	n: f64 = 1
	for i: int = 0; i < mat.n_rows; i += 1 {
		for j: int = 0; j < mat.n_cols; j += 1 {
			mget(mat, i, j)^ = n
			n += 1
		}
	}
}
make_matrix :: proc(mat: ^Matrix, m, n: int, allocator := context.allocator) {
	context.allocator = allocator
	n_elm := int(m) * int(n)
	mat.base = cast([^]f64)mem.alloc(n_elm * size_of(f64))
	mat.n_rows = m
	mat.n_cols = n
}

microkernel_4x4_packed :: proc(k: int, mp_a, mp_b: [^]f64, c: ^Matrix_Slice) #no_bounds_check {
	gamma_0123_c0 := load_4x64(get(c, 0, 0))
	gamma_0123_c1 := load_4x64(get(c, 0, 1))
	gamma_0123_c2 := load_4x64(get(c, 0, 2))
	gamma_0123_c3 := load_4x64(get(c, 0, 3))
	beta_p_j: simd.f64x4
	// iters on cols of A and rows of B:
	for p := 0; p < k; p += 1 {
		alpha_0123_p := load_4x64(mp_a)
		//Axpy Operations:
		beta_p_j = broadcast_4x64(mp_b[0]) // this seeems wrong..?
		gamma_0123_c0 += simd.fma(alpha_0123_p, beta_p_j, gamma_0123_c0)

		beta_p_j = broadcast_4x64(mp_b[1])
		gamma_0123_c1 += simd.fma(alpha_0123_p, beta_p_j, gamma_0123_c1)

		beta_p_j = broadcast_4x64(mp_b[2])
		gamma_0123_c2 += simd.fma(alpha_0123_p, beta_p_j, gamma_0123_c2)

		beta_p_j = broadcast_4x64(mp_b[3])
		gamma_0123_c3 += simd.fma(alpha_0123_p, beta_p_j, gamma_0123_c3)
	}
	store_4x64(get(c, 0, 0), gamma_0123_c0)
	store_4x64(get(c, 0, 1), gamma_0123_c1)
	store_4x64(get(c, 0, 2), gamma_0123_c2)
	store_4x64(get(c, 0, 3), gamma_0123_c3)
}

load_4x64 :: #force_inline proc(src: [^]f64) -> simd.f64x4 {
	return simd.f64x4{src[0], src[1], src[2], src[3]}
}
broadcast_4x64 :: #force_inline proc(v: f64) -> simd.f64x4 {
	return simd.f64x4{v, v, v, v}
}
store_4x64 :: #force_inline proc(dst: [^]f64, mmx: simd.f64x4) {
	dst[0] = simd.extract(mmx, 0)
	dst[1] = simd.extract(mmx, 1)
	dst[2] = simd.extract(mmx, 2)
	dst[3] = simd.extract(mmx, 3)
}

mmult :: proc(A: ^Matrix, B: ^Matrix, C: ^Matrix, allocator := context.allocator) #no_bounds_check {
	m := A.n_rows
	n := B.n_cols
	k := A.n_cols
	assert(k == B.n_rows)
	m_c :: 96
	n_c :: 96
	k_c :: 96
	n_r :: 4
	m_r :: 4
	// Memory Objectives: 
	// Packed Panel ~B_pj in L3 Cache
	// Packed Tile ~A_ip in L2 Cache
	// Micro-Panel j of ~B_pj in L1 Cache
	// Micro-Tile C_ij_ji in mmx registers
	bytes_a := k_c * m_c * size_of(f64)
	bytes_b := k_c * n_c * size_of(f64)
	a_pack := cast([^]f64)mem.alloc(bytes_a)
	b_pack := cast([^]f64)mem.alloc(bytes_b)
	defer mem.free(a_pack)
	defer mem.free(b_pack)
	fmt.println("A&B Cache-Packing Temp Allocs (kb):", bytes_a / 1024, bytes_b / 1024)

	for j_iter := 0; j_iter < n; j_iter += n_c {
		// Loop 5: Slice B & C into Column-Panels C_j & B_j (width n_c)
		j := min(n_c, n - j_iter) // last tile can be < block size
		//
		for p_iter := 0; p_iter < k; p_iter += k_c {
			//Loop 4: Slice A into Column-Panel A_p (width k_c), Slice B_j into row-panels B_pj (height k_c)
			// Pack B_pj into ~B_pj. Packed Column width is n_r
			p := min(k_c, k - p_iter)
			B_pj := get_slice(B, p, j)
			pack_by_cols(b_pack, &B_pj, n_c, k_c)
			//
			for i_iter := 0; i_iter < m; i_iter += m_c {
				// Loop 3: Slice Panel A_p by rows into A_ip (height m_c);
				// Pack A_ip into ~A_ip; Slice C_j into row-panels C_ij (height m_c) Packed row-height is m_r
				i := min(m_c, m - i_iter)
				A_ip := get_slice(A, i, p)
				C_ij := get_slice(C, i, j)
				pack_by_rows(a_pack, &A_ip, i, k_c)
				//
				gemm_ij_kernel(i, j, p, a_pack, b_pack, &C_ij)
			}
		}
	}
}

gemm_ij_kernel :: proc(m, n, k: int, a_tilde, b_tilde: [^]f64, c_ij: ^Matrix_Slice) #no_bounds_check {
	n_r :: 4
	m_r :: 4
	for j_iter := 0; j_iter < n; j_iter += n_r {
		//Loop 2: Slice C_ij into micro panels (width: n_r)
		j := min(n_r, n - j_iter) // todo: remove in favor of asserts?
		//
		for i_iter := 0; i_iter < m; i_iter += m_r {
			//Loop 1: Slice C_ij into a tile with height: m_r, Slice b_tilde by i
			i := min(m_r, m - i_iter)
			a_tile := transmute([^]f64)&a_tilde[i * k]
			b_micro_panel := transmute([^]f64)&b_tilde[j * k]
			c_micro_tile := get_slice(c_ij, i, j)
			microkernel_4x4_packed(k, a_tile, b_micro_panel, &c_micro_tile)
		}
	}
}

pack_by_cols :: proc(dst: [^]f64, src: ^Matrix_Slice, n, k: int) #no_bounds_check {
	//K:Rows - N:Cols
	n_r :: 4
	// Pack a KC x NC panel of B.  NC is assumed to be a multiple of NR.  The block is 
	// packed into Btilde a micro-panel at a time. If necessary, the last micro-panel 
	// is padded with columns of zeroes.
	full_width := n == n_r
	iter := 0
	for c_iter := 0; c_iter < n; c_iter += n_r {
		c := min(n_r, n - c_iter)
		col := get_slice(src, 0, c)
		if full_width {
			for p := 0; p < k; p += 1 {
				for j in 0 ..< n_r {
					dst[iter] = get(&col, p, j)^
					iter += 1
				}
			}
		} else {
			for p := 0; p < k; p += 1 {
				for j in 0 ..< n_r {
					dst[iter] = get(&col, p, j)^
					iter += 1
				}
				for j := c; j < n_r; j += 1 {
					dst[iter] = 0.
					iter += 1
				}
			}
		}
	}
}
pack_by_rows :: proc(dst: [^]f64, src: ^Matrix_Slice, m, k: int) #no_bounds_check {
	//K:cols - N:rows
	m_r :: 4
	// Pack a MC x KC block of A.  MC is assumed to be a multiple of MR.  The block is 
	// packed into Atilde a micro-panel at a time. If necessary, the last micro-panel 
	// is padded with rows of zeroes.
	assert(m % m_r == 0, "partial rows not impl")
	iter := 0
	for r_iter := 0; r_iter < m; r_iter += m_r {
		r := min(m_r, m - r_iter)
		row := get_slice(src, r, 0)
		for p := 0; p < k; p += 1 {
			for i in 0 ..< m_r {
				dst[iter] = get(&row, i, p)^
				iter += 1
			}
		}

	}
}

import "core:math/rand"
import "core:intrinsics"
import "core:fmt"
@(test)
perf_test :: proc(t: ^testing.T) {
	fmt.println("AMD 3950X: L1: 1mb, L2: 8mb, L3:64mb (shared), GFLOPS/Core: 225, Clock: 3.5 gHz (4.7 Turbo)")
	fmt.println("Expected Performance: ~90% Max, 200 GFLOP")
	//
	m: int = 96 * 10
	n_elm := m * m
	mb_ea := f64(n_elm) / 1024. / 1024.
	fmt.println("Per-Matrix Size:", mb_ea, "mb")
	//
	A: Matrix
	make_matrix(&A, m, m)
	defer mem.free(A.base)
	// set_random(&A)
	set_ones(&A)

	B: Matrix
	make_matrix(&B, m, m)
	defer mem.free(B.base)
	set_random(&B)
	// set_n(&B)

	C: Matrix
	make_matrix(&C, m, m)
	mem.zero(C.base, int(C.n_rows * C.n_cols) * size_of(f64))
	defer mem.free(C.base)

	fmt.println("starting mmult")
	start := intrinsics.read_cycle_counter()
	//mmult_jpi(&A, &B, &C) // Naive Impl
	mmult(&A, &B, &C) // Cache-Aware SIMD impl
	end := intrinsics.read_cycle_counter()
	n_flops := 2 * m * m * m
	clocks := end - start
	time := f64(clocks) / 3.5 // ns
	fmt.printf("Clocks: %d, n_flops: %d, time(ms):%f, GFS:%f\n", end - start, n_flops, time / 1E6, (f64(n_flops)) / time)
	fmt.println("clocks/flop", f64(end - start) / f64(n_flops))
	fmt.println("END TEST")
}
