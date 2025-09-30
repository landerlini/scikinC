from scikinC.layers.BaseLayerConverter import BaseLayerConverter
from scikinC._tools import array2c 

class Dense (BaseLayerConverter):
  """
  Dense Layer converter
  """

  def definition(self):
    """Return the definition of the layer function"""
    nX, nY = self.layer.kernel.shape
    kernel, bias = self.layer.get_weights()
    c_code = """
#if defined(USE_AVX2_32) || defined(USE_AVX2_64)
  #include <immintrin.h>
  #include <string.h>
#endif

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

    extern "C"
    FLOAT_T* %(layername)s (FLOAT_T* ret, const FLOAT_T* input)
    {
      int i, j, ii, jj;
      static const FLOAT_T kernel[%(nY)d][%(nX)d] = %(kernel_values)s;
      static const FLOAT_T bias[%(nY)d] = %(bias_values)s;

      // Block sizes 
      const int BLOCK_I = 32;
      const int BLOCK_J = CACHE_LINE_SIZE / sizeof(FLOAT_T);


#if defined(USE_AVX2_32)
      const int word_size = 8; // 256 bits / 32 bits
      memcpy(ret, bias, sizeof(FLOAT_T)*%(nY)d);

      // Blocked scalar version for float with AVX2
      for (ii = 0; ii < %(nY)d; ii += BLOCK_I) {
        int i_max = (ii + BLOCK_I < %(nY)d) ? ii + BLOCK_I : %(nY)d;
        for (jj = 0; jj < %(nX)d; jj += BLOCK_J) {
          const int j_max = (jj + BLOCK_J < %(nX)d) ? jj + BLOCK_J : %(nX)d;
          for (i = ii; i < i_max; ++i) {
            __m256 sum = _mm256_setzero_ps();
            for (j = jj; j + word_size <= j_max; j += word_size) {
              __m256 in_vec = _mm256_loadu_ps(&input[j]);
              __m256 ker_vec = _mm256_loadu_ps(&kernel[i][j]);
              sum = _mm256_fmadd_ps(in_vec, ker_vec, sum);
            }

            float temp[word_size];
            _mm256_storeu_ps(temp, sum);
            for (int k = 0; k < word_size; ++k) 
              ret[i] += temp[k];  

            // Scalar tail
            for (; j < j_max; ++j) {
              ret[i] += input[j] * kernel[i][j];
            }
          }
        }
      }
#elif defined(USE_AVX2_64)
      const int word_size = 4; // 256 bits / 64 bits
      memcpy(ret, bias, sizeof(FLOAT_T)*%(nY)d);

      // Blocked scalar version for double with AVX2
      for (ii = 0; ii < %(nY)d; ii += BLOCK_I) {
        int i_max = (ii + BLOCK_I < %(nY)d) ? ii + BLOCK_I : %(nY)d;
        for (jj = 0; jj < %(nX)d; jj += BLOCK_J) {
          const int j_max = (jj + BLOCK_J < %(nX)d) ? jj + BLOCK_J : %(nX)d;
          for (i = ii; i < i_max; ++i) {
            __m256d sum = _mm256_setzero_pd();
            for (j = jj; j + word_size <= j_max; j += word_size) {
              __m256d in_vec = _mm256_loadu_pd(&input[j]);
              __m256d ker_vec = _mm256_loadu_pd(&kernel[i][j]);
              sum = _mm256_fmadd_pd(in_vec, ker_vec, sum);
            }

            double temp[word_size];
            _mm256_storeu_pd(temp, sum);
            for (int k = 0; k < word_size; ++k) 
              ret[i] += temp[k];  

            // Scalar tail
            for (; j < j_max; ++j) {
              ret[i] += input[j] * kernel[i][j];
            }
          }
        }
      }
#else
      for (i = 0; i < %(nY)d; ++i)
        ret[i] = bias[i];

      // Blocked scalar version (used for double or if AVX2 is not enabled)
      for (ii = 0; ii < %(nY)d; ii += BLOCK_I) {
        int i_max = (ii + BLOCK_I < %(nY)d) ? ii + BLOCK_I : %(nY)d;
        for (jj = 0; jj < %(nX)d; jj += BLOCK_J) {
          int j_max = (jj + BLOCK_J < %(nX)d) ? jj + BLOCK_J : %(nX)d;
          for (i = ii; i < i_max; ++i) {
            for (j = jj; j < j_max; ++j) {
              ret[i] += input[j] * kernel[i][j];
            }
          }
        }
      }
#endif

      for (i = 0; i < %(nY)d; ++i) {
        %(activate)s
      }

      return ret;
    }
    """ % dict(
      layername=self.name,
      nX=nX,
      nY=nY,
      kernel_values=array2c(kernel.T),
      bias_values=array2c(bias),
      activate=self.activate('ret[i]'),
    )
    return c_code

  def call(self, obuffer, ibuffer):
    """Return the call to the layer function""" 
    return "%(layername)s ( %(obuffer)s, %(ibuffer)s);" % dict (
        layername=self.name, obuffer=obuffer, ibuffer=ibuffer )
