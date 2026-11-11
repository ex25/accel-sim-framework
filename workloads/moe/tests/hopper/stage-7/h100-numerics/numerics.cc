// Offline checks of the production arithmetic against captured GPU outputs.
#include "hopper_math.h"
#include "nvidia_math.h"

extern "C" void math_ops(unsigned count, const float *input, uint32_t *output) {
  for (unsigned i = 0; i < count; ++i) {
    float a = input[2 * i], b = input[2 * i + 1];
    output[5 * i] = nvidia_math::bits(nvidia_math::ex2(a, false));
    output[5 * i + 1] = nvidia_math::bits(nvidia_math::rcp(b, false));
    output[5 * i + 2] =
        nvidia_math::bits(nvidia_math::divide(1.f, b, true, false));
    output[5 * i + 3] = nvidia_math::bits(1.f / b);
    output[5 * i + 4] =
        nvidia_math::bits(nvidia_math::divide(1.f, b, false, false));
  }
}

extern "C" void general_math_ops(unsigned count, const float *input,
                                 uint32_t *output) {
  for (unsigned i = 0; i < count; ++i) {
    float a = input[2 * i], b = input[2 * i + 1];
    output[9 * i] = nvidia_math::bits(nvidia_math::ex2(a, false));
    output[9 * i + 1] = nvidia_math::bits(nvidia_math::ex2(a, true));
    output[9 * i + 2] = nvidia_math::bits(nvidia_math::rcp(b, false));
    output[9 * i + 3] = nvidia_math::bits(nvidia_math::rcp(b, true));
    output[9 * i + 4] =
        nvidia_math::bits(nvidia_math::divide(a, b, true, false));
    output[9 * i + 5] =
        nvidia_math::bits(nvidia_math::divide(a, b, true, true));
    output[9 * i + 6] =
        nvidia_math::bits(nvidia_math::divide(a, b, false, false));
    output[9 * i + 7] =
        nvidia_math::bits(nvidia_math::divide(a, b, false, true));
    output[9 * i + 8] = nvidia_math::bits(a / b);
  }
}

extern "C" void activation_stages(unsigned count, const float *input,
                                  uint32_t *output) {
  for (unsigned i = 0; i < count; ++i) {
    float g = input[2 * i], u = input[2 * i + 1];
    float x = -g * 0x1.715476p+0f;
    float e = nvidia_math::ex2(x, false), den = 1.f + e;
    float inv = nvidia_math::divide(1.f, den, true, false);
    float gate = g * inv, product = gate * u;
    uint32_t *row = output + 8 * i;
    row[0] = nvidia_math::bits(x);
    row[1] = nvidia_math::bits(e);
    row[2] = nvidia_math::bits(den);
    row[3] = nvidia_math::bits(inv);
    row[4] = nvidia_math::bits(gate);
    row[5] = nvidia_math::bits(product);
    uint32_t bits = row[5];
    row[6] = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
    row[7] = nvidia_math::bits(g);
  }
}

extern "C" void blocks(unsigned count, const uint16_t *a, const uint16_t *b,
                       const float *c, float *out) {
  for (unsigned i = 0; i < count; ++i)
    out[i] = hopper_bf16_block(a + 16 * i, b + 16 * i, c[i]);
}

extern "C" void dot(unsigned rows, unsigned cols, const float *w,
                    const float *x, float *out, float *raw) {
  for (unsigned i = 0; i < rows; ++i) {
    float sum = 0;
    for (unsigned k = 0; k < cols; k += 16) {
      uint16_t a[16], b[16];
      for (unsigned j = 0; j < 16; ++j) {
        a[j] = nvidia_math::bits(w[i * cols + k + j]) >> 16;
        b[j] = nvidia_math::bits(x[k + j]) >> 16;
      }
      sum = hopper_bf16_block(a, b, sum);
    }
    raw[i] = sum;
    uint32_t bits = nvidia_math::bits(sum);
    out[i] =
        nvidia_math::value((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
  }
}
