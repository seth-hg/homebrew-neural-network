#include "mlp.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <omp.h>
#if __aarch64__
#include <arm_neon.h>
#elif __AVX2__
#include <immintrin.h>
#endif

void softmax(size_t n, float *input) {
  float sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    input[i] = exp(input[i]);
    sum += input[i];
  }
  sum = 1 / sum;
  for (size_t i = 0; i < n; ++i) {
    input[i] = input[i] * sum;
  }
}

// https://github.com/pytorch/pytorch/blob/caea1adc35404b39fe4f2e3fa75f9a0b9e554bbc/aten/src/ATen/native/SoftMax.cpp#L15
void log_softmax(size_t n, float *input) {
  float max_c = input[0];
  for (size_t i = 1; i < n; ++i) {
    max_c = std::max(max_c, input[i]);
  }
  float sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum += exp(input[i] - max_c);
  }
  sum = log(sum);
  for (size_t i = 0; i < n; ++i) {
    input[i] = input[i] - max_c - sum;
  }
}

void relu(size_t n, float *x) {
  for (size_t i = 0; i < n; ++i) {
    x[i] = x[i] > 0 ? x[i] : 0.0;
  }
}

size_t argmax(size_t n, const float *values) {
  size_t max_idx = 0;
  float max_value = values[0];
  for (size_t i = 1; i < n; ++i) {
    if (values[i] > max_value) {
      max_value = values[i];
      max_idx = i;
    }
  }
  return max_idx;
}

void Layer::forwardBatch(size_t n, const std::vector<float> &input, std::vector<float> *output) const {
  if (input.size() < n * in_features_) {
    throw std::runtime_error("input shape mismatch");
  }

  // input: n * in_features
  // weight: out_features * in_features
  // output: n * out_features
  // output = input * transpose(weight) + bias
  output->resize(n * out_features_);
#pragma omp parallel for
  for (size_t k = 0; k < n; ++k) {
    for (size_t i = 0; i < out_features_; ++i) {
      const float *x = input.data() + k * in_features_;
      const float *y = weight_.data() + i * in_features_;
      size_t j = in_features_;
      float sum = 0.0;
#if __aarch64__
      float buf[4] = {0, 0, 0, 0};
      float32x4_t msum = vld1q_f32(buf); // initialize msum to <0,0,0,0>

      while (j >= 4) {
        float32x4_t mx = vld1q_f32(x);
        float32x4_t my = vld1q_f32(y);
        msum = vmlaq_f32(msum, mx, my);
        x += 4;
        y += 4;
        j -= 4;
      }

      vst1q_f32(buf, msum);
      sum = buf[0] + buf[1] + buf[2] + buf[3];
#elif __AVX2__
      __m256 msum1 = _mm256_setzero_ps();

      while (j >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        __m256 my = _mm256_loadu_ps(y);
        msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(mx, my));
        x += 8;
        y += 8;
        j -= 8;
      }

      __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
      msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));

      if (j >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        __m128 my = _mm_loadu_ps(y);
        msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
        x += 4;
        y += 4;
        j -= 4;
      }

      msum2 = _mm_hadd_ps(msum2, msum2);
      msum2 = _mm_hadd_ps(msum2, msum2);
      sum = _mm_cvtss_f32(msum2);
#endif
      while (j > 0) {
        sum += (*x) * (*y);
        x += 1;
        y += 1;
        j -= 1;
      }
      (*output)[k * out_features_ + i] = (sum + bias_[i]);
    }
  }
}

void MLP::forwardBatch(size_t n, const std::vector<float> &input, std::vector<float> *output) const {
  std::vector<float> layer_input;
  std::vector<float> layer_output(input.begin(), input.end());
  // hidden layers
  size_t i = 0;
  for (; i < layers_.size() - 1; ++i) {
    layer_input.swap(layer_output);
    layers_[i].forwardBatch(n, layer_input, &layer_output);
    relu(layer_output.size(), layer_output.data());
    layer_input.clear();
  }
  // output layer, without ReLU activation
  layer_input.swap(layer_output);
  layers_[i].forwardBatch(n, layer_input, &layer_output);
  // softmax
  for (size_t i = 0; i < n; ++i) {
    log_softmax(out_features_, layer_output.data() + i * out_features_);
  }
  output->swap(layer_output);
}

int MLP::classify(const std::vector<float> &feature) const {
  std::vector<float> output;
  forward(feature, &output);
  return argmax(output.size(), output.data());
}

void MLP::classify(size_t n, const std::vector<float> &features, std::vector<int> *classes) const {
  std::vector<float> output;
  forwardBatch(n, features, &output);
  classes->resize(n);
#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    size_t idx = argmax(out_features_, output.data() + out_features_ * i);
    (*classes)[i] = idx;
  }
}

bool MLP::load(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary);
  std::string magic(4, 0);
  ifs.read(magic.data(), 4);
  if (magic != "HBNN") {
    return false;
  }

  uint32_t n_layers;
  ifs.read(reinterpret_cast<char *>(&n_layers), 4);
  if (n_layers != layers_.size()) {
    return false;
  }

  for (uint32_t i = 0; i < n_layers; ++i) {
    auto shape = layers_[i].shape();
    // load weight
    uint32_t rows = 0, cols = 0;
    ifs.read(reinterpret_cast<char *>(&rows), sizeof(uint32_t));
    ifs.read(reinterpret_cast<char *>(&cols), sizeof(uint32_t));
    if (rows != shape.first || cols != shape.second) {
      return false;
    }

    ifs.read(reinterpret_cast<char *>(layers_[i].weight_.data()), rows * cols * sizeof(float));

    // load bias
    ifs.read(reinterpret_cast<char *>(&rows), sizeof(uint32_t));
    if (rows != layers_[i].bias_.size()) {
      return false;
    }
    if (rows == 0) {
      continue;
    }
    ifs.read(reinterpret_cast<char *>(layers_[i].bias_.data()), rows * sizeof(float));
  }

  return true;
}
