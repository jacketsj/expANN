// External
#include <immintrin.h>
#include <math.h>

#include <Eigen/Dense>

inline __m256i load_128bit_to_256bit(const __m128i* ptr) {
	__m128i value128 = _mm_loadu_si128(ptr);
	__m256i value256 = _mm256_castsi128_si256(value128);
	return _mm256_inserti128_si256(value256, _mm_setzero_si128(), 1);
}
inline int32_t distance_compare_avx512f_i32(const int16_t* vec1,
																						const int16_t* vec2, size_t size) {
	__m512i sum_squared_diff = _mm512_setzero_si512();

	for (size_t i = 0; i < size; i += 32) {
		__m512i a = _mm512_loadu_si512((__m512i*)&vec1[i]);
		__m512i b = _mm512_loadu_si512((__m512i*)&vec2[i]);
		__m512i diff = _mm512_sub_epi16(a, b);
		__m512i sqr_diff = _mm512_mullo_epi16(diff, diff);
		sum_squared_diff = _mm512_add_epi32(
				sum_squared_diff, _mm512_madd_epi16(_mm512_set1_epi16(1), sqr_diff));
	}
	return _mm512_reduce_add_epi32(sum_squared_diff);
}

inline int32_t distance_compare_avx512f_i64(const int8_t* vec1,
																						const int8_t* vec2, size_t size) {
	__m512i sum_squared_diff = _mm512_setzero_si512();

	for (size_t i = 0; i < size; i += 64) { // Process 64 elements at a time
		__m512i a = _mm512_loadu_si512((__m512i*)&vec1[i]);
		__m512i b = _mm512_loadu_si512((__m512i*)&vec2[i]);
		__m512i diff = _mm512_sub_epi8(a, b);

		// Extend 8-bit differences to 16-bit by zero extension
		__m512i diff16_lo = _mm512_unpacklo_epi8(diff, _mm512_setzero_si512());
		__m512i diff16_hi = _mm512_unpackhi_epi8(diff, _mm512_setzero_si512());

		// Square the 16-bit differences and produce 32-bit integers directly
		__m512i sqr_diff_lo = _mm512_madd_epi16(diff16_lo, diff16_lo);
		__m512i sqr_diff_hi = _mm512_madd_epi16(diff16_hi, diff16_hi);

		// Sum the lower and higher squared differences as 32-bit integers
		sum_squared_diff = _mm512_add_epi32(sum_squared_diff, sqr_diff_lo);
		sum_squared_diff = _mm512_add_epi32(sum_squared_diff, sqr_diff_hi);
	}

	// Reduce across the vector to get a single sum
	return _mm512_reduce_add_epi32(sum_squared_diff);

	/*
Eigen::Map<const Eigen::Array<int8_t, Eigen::Dynamic, 1>> mappedVec1(vec1,
																																		 size);
Eigen::Map<const Eigen::Array<int8_t, Eigen::Dynamic, 1>> mappedVec2(vec2,
																																		 size);

return (mappedVec1.cast<int>() - mappedVec2.cast<int>()).square().sum();
*/
	// __m512i sum_squared_diff = _mm512_setzero_si512();

	// for (size_t i = 0; i < size; i += 64) { // Process 64 elements at a time
	// 	__m512i a = _mm512_loadu_si512((__m512i*)&vec1[i]);
	// 	__m512i b = _mm512_loadu_si512((__m512i*)&vec2[i]);
	// 	__m512i diff = _mm512_sub_epi8(a, b);
	// 	// Convert diff to 16-bit by multiplying by 1
	// 	__m512i diff16 = _mm512_maddubs_epi16(_mm512_set1_epi8(1), diff);
	// 	// Square the differences and convert to 32-bit
	// 	__m512i sqr_diff = _mm512_madd_epi16(diff16, diff16);
	// 	sum_squared_diff = _mm512_add_epi32(sum_squared_diff, sqr_diff);
	// }
	// // Reduce across the vector to get a single sum
	// return _mm512_reduce_add_epi32(sum_squared_diff);
	/*
// Cast to int to prevent overflow and compute the squared differences
Eigen::Array<int, Eigen::Dynamic, 1> diff =
	(mappedVec1.cast<int>() - mappedVec2.cast<int>()).square();

// Sum all the squared differences
return diff.sum();
*/
}

inline float distance_compare_avx512f_f16(const float* vec1, const float* vec2,
																					size_t size) {

	__m512 sum_squared_diff = _mm512_setzero_ps();

	for (size_t i = 0; i < size; i += 16) {
		__m512 a = _mm512_loadu_ps(&vec1[i]);
		__m512 b = _mm512_loadu_ps(&vec2[i]);
		__m512 diff = _mm512_sub_ps(a, b);
		sum_squared_diff = _mm512_fmadd_ps(diff, diff, sum_squared_diff);
	}
	return _mm512_reduce_add_ps(sum_squared_diff);

	/*
	{
		__m512 sum_squared_diff = _mm512_setzero_ps();

		for (int i = 0; i < size / 16; i += 1) {
			__m512 v1 = _mm512_cvtph_ps(
					_mm256_loadu_si256((const __m256i*)(vec1 + i * 2 * 16)));
			__m512 v2 = _mm512_cvtph_ps(
					_mm256_loadu_si256((const __m256i*)(vec2 + i * 2 * 16)));

			__m512 diff = _mm512_sub_ps(v1, v2);
			sum_squared_diff = _mm512_fmadd_ps(diff, diff, sum_squared_diff);
		}

		// size_t i = (size / 16) * 16;

		// if (i != size) {
		//	__m512 va =
		//			_mm512_cvtph_ps(load_128bit_to_256bit((const __m128i*)(vec1 + i *
		// 2)));
		//	__m512 vb =
		//			_mm512_cvtph_ps(load_128bit_to_256bit((const __m128i*)(vec2 + i *
		// 2)));
		//	__m512 diff512 = _mm512_sub_ps(va, vb);
		//	sum_squared_diff = _mm512_fmadd_ps(diff512, diff512, sum_squared_diff);
		// }

		return _mm512_reduce_add_ps(sum_squared_diff);
	}
	*/
}

inline float dot_avx512f_f16(const float* vec1, const float* vec2,
														 size_t size) {
	__m512 sum = _mm512_setzero_ps();
	for (size_t i = 0; i < size; i += 16) {
		__m512 a = _mm512_loadu_ps(&vec1[i]);
		__m512 b = _mm512_loadu_ps(&vec2[i]);
		sum = _mm512_fmadd_ps(a, b, sum);
	}
	return _mm512_reduce_add_ps(sum);
}
