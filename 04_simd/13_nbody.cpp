#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

float vecsum(__m256 vec) {
  const __m256 qsum = _mm256_permute_ps(_mm256_hadd_ps(vec, vec), 0b11010100); /* x7+x6, x5+x4, _, _, x3+x2, x1+x0, _, _ */
  const __m256 dsum = _mm256_hadd_ps(qsum, qsum);
  const __m128 high = _mm256_extractf128_ps(dsum, 1); /* cut in half */
  const __m128 low = _mm256_castps256_ps128(dsum);
  const __m128 sum = _mm_add_ps(high, low);
  return _mm_cvtss_f32(sum);
}

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  for(int i=0; i<N; i++) {
    __m256 mi = _mm256_set1_ps(m[i]);
    __m256 mvec = _mm256_load_ps(m);
    __m256 mask = _mm256_cmp_ps(mvec, mi, _CMP_NEQ_OQ);
    __m256 mk = _mm256_blendv_ps(mask, _mm256_set1_ps(1), mask);
    mvec = _mm256_mul_ps(mvec, mask);
    __m256 rx = _mm256_set1_ps(x[i]);
    __m256 ry = _mm256_set1_ps(y[i]);
    rx = _mm256_blendv_ps(rx, _mm256_sub_ps(rx, xvec), mask);
    ry = _mm256_blendv_ps(ry, _mm256_sub_ps(ry, yvec), mask);
    __m256 r = _mm256_add_ps(_mm256_mul_ps(rx, rx), _mm256_mul_ps(ry, ry));
    r = _mm256_rsqrt_ps(r);
    __m256 fxvec = _mm256_mul_ps(rx, _mm256_mul_ps(_mm256_mul_ps(mvec, r), _mm256_mul_ps(r, r)));
    __m256 fyvec = _mm256_mul_ps(ry, _mm256_mul_ps(_mm256_mul_ps(mvec, r), _mm256_mul_ps(r, r)));
    fx[i] = -vecsum(fxvec);
    fy[i] = -vecsum(fyvec);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
