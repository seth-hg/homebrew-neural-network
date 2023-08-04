#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

class Layer {
public:
  Layer(size_t in_features, size_t out_features)
      : in_features_(in_features), out_features_(out_features), weight_(in_features * out_features),
        bias_(out_features) {
    // TODO: initialize with random weights
  }

  void forward(const std::vector<float> &input, std::vector<float> *output) const { forwardBatch(1, input, output); }

  void forwardBatch(size_t n, const std::vector<float> &input, std::vector<float> *output) const;

  std::pair<size_t, size_t> shape() const { return std::make_pair(out_features_, in_features_); }

  friend class MLP;

private:
  size_t in_features_;
  size_t out_features_;
  std::vector<float> weight_;
  std::vector<float> bias_;
};

class MLP {
public:
  MLP(std::initializer_list<size_t> sizes) {
    auto last = sizes.begin();
    in_features_ = *last;
    for (auto it = last + 1; it != sizes.end(); ++it) {
      out_features_ = *it;
      layers_.emplace_back(*last, *it);
      last = it;
    }
  }

  void forward(const std::vector<float> &input, std::vector<float> *output) const { forwardBatch(1, input, output); }

  void forwardBatch(size_t n, const std::vector<float> &input, std::vector<float> *output) const;

  int classify(const std::vector<float> &feature) const;

  void classify(size_t n, const std::vector<float> &features, std::vector<int> *classes) const;

  bool load(const std::string &path);

private:
  size_t in_features_;
  size_t out_features_;
  std::vector<Layer> layers_;
};
